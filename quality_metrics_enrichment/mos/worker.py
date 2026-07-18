"""MOS assessment worker — orchestrates UTMOS / SQUIM / DNSMOS Pro / AudioBox.

Each metric is opt-in via cfg flags. Per-batch H2D runs on a dedicated
transfer stream so the next batch's upload overlaps with the current
batch's d2h DMA. DNSMOS Pro runs on a side stream so its launch-bound
kernels can slip in alongside the bigger SSL forwards.
"""

from __future__ import annotations

import contextlib
import json
import logging
import signal
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from common.loader import SharAudioLoader
from common.timing import RunMetrics, StageTimer

from .models.audiobox import (
    DEFAULT_SUB_BATCH as _DEFAULT_AUDIOBOX_SUB_BATCH,
    WIN_SEC as _AUDIOBOX_WIN_SEC,
    build_audiobox,
)
from .models.dnsmos import (
    DEFAULT_VARIANT as _DEFAULT_DNSMOS_VARIANT,
    STFT_HOP as _STFT_HOP,
    STFT_N_FFT as _STFT_N_FFT,
    STFT_WIN as _STFT_WIN,
    build_dnsmos,
)
from .models.squim import build_squim
from .models.utmos import build_utmos
from .pending import PendingBatch
from .profiler import load_seen_cut_ids, make_profiler

logger = logging.getLogger(f"pipeline.{__name__}")

_DEFAULT_TRUNCATE_SECS = 10.0

STAGES: tuple[str, ...] = (
    "mos/h2d_transfer",
    "mos/gpu_compute",
    "mos/d2h_post",
    "mos/d2h_sync",
    "mos/jsonl_write",
)


class MosAssessmentWorker:
    """Runs the enabled MOS metrics on Lhotse Shar audio. One JSONL per rank."""

    def __init__(
        self,
        cfg: dict,
        rank: int,
        world_size: int,
        local_rank: int = 0,
        decode_num_workers: int = 4,
        metrics: Optional[RunMetrics] = None,
    ) -> None:
        self._cfg = cfg
        self._rank = rank
        self._device = torch.device(f"cuda:{local_rank}")
        self._dtype = torch.bfloat16
        self._truncate_secs = float(cfg.get("truncate_secs", _DEFAULT_TRUNCATE_SECS))
        self._audiobox_sub_batch = int(
            cfg.get("audiobox_sub_batch", _DEFAULT_AUDIOBOX_SUB_BATCH)
        )
        self._world_size = world_size
        self._stop_requested = False
        self._metrics = metrics or RunMetrics(
            rank=rank, world_size=world_size,
            pipeline="mos", run_name="adhoc",
            stages=STAGES,
        )

        self._use_utmos = bool(cfg.get("use_utmos", True))
        self._use_squim = bool(cfg.get("use_squim", True))
        self._use_dnsmos = bool(cfg.get("use_dnsmos", True))
        self._use_audiobox = bool(cfg.get("use_audiobox", False))
        self._fail_fast_oom = bool(cfg.get("fail_fast_oom", False))
        self._dnsmos_variant = str(
            cfg.get("dnsmos_variant", _DEFAULT_DNSMOS_VARIANT)
        ).lower()

        self._loader = SharAudioLoader(
            cfg, rank=rank, world_size=world_size,
            num_workers=decode_num_workers,
            prefetch_factor=int(cfg.get("prefetch_factor", 6)),
            dataloader_timeout=int(cfg.get("dataloader_timeout", 300)),
        )

        active = []
        if self._use_utmos:
            self._utmos = build_utmos(cfg, self._device)
            active.append("UTMOS")
        if self._use_squim:
            self._squim = build_squim(cfg, self._device)
            active.append("SQUIM")
        if self._use_dnsmos:
            self._dnsmos = build_dnsmos(cfg, self._device)
            self._stft_window = torch.hann_window(_STFT_WIN, device=self._device)
            active.append(f"DNSMOS-{self._dnsmos_variant}")
        if self._use_audiobox:
            self._audiobox = build_audiobox(cfg)
            active.append("AudioBox")

        # DNSMOS Pro (~0.07M params) is launch-bound — a side stream lets
        # its kernels slip in alongside the bigger SSL forwards.
        self._dnsmos_stream: Optional[torch.cuda.Stream] = (
            torch.cuda.Stream(device=self._device) if self._use_dnsmos else None
        )
        # Dedicated H2D stream for double-buffered uploads.
        self._transfer_stream = torch.cuda.Stream(device=self._device)

        self._install_signal_handlers()
        logger.info(
            "MosAssessmentWorker ready on %s | active: %s | truncate=%.1fs | audiobox_sub_batch=%d",
            torch.cuda.get_device_name(self._device),
            ", ".join(active), self._truncate_secs, self._audiobox_sub_batch,
        )

    def _install_signal_handlers(self) -> None:
        def _handle(signum, _frame):
            if not self._stop_requested:
                logger.warning(
                    "Received %s — finishing current batch and exiting.",
                    signal.Signals(signum).name,
                )
            self._stop_requested = True

        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(sig, _handle)
            except (ValueError, OSError):
                pass

    def _dnsmos_spec(self, audio: torch.Tensor) -> torch.Tensor:
        """Log-magnitude STFT, reshaped to ``(B, 1, frames, F)`` for DNSMOS Pro."""
        spec = torch.stft(
            audio,
            n_fft=_STFT_N_FFT,
            hop_length=_STFT_HOP,
            win_length=_STFT_WIN,
            window=self._stft_window,
            return_complex=True,
        ).abs()
        spec = spec.clamp_(min=1e-7).log10_()
        return spec.transpose(-1, -2).unsqueeze(1)

    def _audiobox_compute(
        self,
        audio_gpu: torch.Tensor,
        T: int,
        sr: int,
        eff_lengths: torch.Tensor,
        _rf,
    ) -> dict:
        """Run AudioBox over fixed-length windows; return per-axis GPU tensors."""
        with _rf("mos/audiobox_compute"):
            win_len = int(_AUDIOBOX_WIN_SEC * sr)
            B_new = audio_gpu.shape[0]

            n_wins = (T + win_len - 1) // win_len
            pad_to = n_wins * win_len
            audio_p = F.pad(audio_gpu, (0, pad_to - T))
            windows = audio_p.view(B_new, n_wins, win_len)

            pos = torch.arange(pad_to, device=self._device)
            valid = pos[None, :] < eff_lengths.to(self._device)[:, None]
            masks = valid.view(B_new, n_wins, win_len)

            weights_2d = masks.float().sum(dim=2) / win_len
            has_content = weights_2d > 0

            wav_flat = windows.reshape(-1, win_len)[has_content.view(-1)]
            mask_flat = masks.reshape(-1, win_len)[has_content.view(-1)]
            w_flat = weights_2d.view(-1)[has_content.view(-1)]
            bid_flat = (
                torch.arange(B_new, device=self._device)[:, None]
                .expand(B_new, n_wins)
                .reshape(-1)[has_content.view(-1)]
            )

            tr = self._audiobox.target_transform
            N_valid = wav_flat.shape[0]
            sub_bs = self._audiobox_sub_batch

            num_acc = {ax: torch.zeros(B_new, device=self._device) for ax in ("CE", "CU", "PC", "PQ")}
            denom_acc = {ax: torch.zeros(B_new, device=self._device) for ax in ("CE", "CU", "PC", "PQ")}

            # Pad N_valid to a multiple of sub_bs so every iter has the same
            # tensor shape — caching allocator reuses the same buffers.
            pad_n = (-N_valid) % sub_bs
            if pad_n:
                _z = torch.zeros(pad_n, win_len, device=self._device)
                wav_flat = torch.cat([wav_flat, _z], dim=0)
                mask_flat = torch.cat([mask_flat, _z.bool()], dim=0)
                w_flat = torch.cat([w_flat, torch.zeros(pad_n, device=self._device)])
                bid_flat = torch.cat([bid_flat, torch.zeros(pad_n, dtype=torch.long, device=self._device)])

            N_padded = N_valid + pad_n
            for sb in range(0, N_padded, sub_bs):
                sb_preds = self._audiobox.model({
                    "wav": wav_flat[sb:sb + sub_bs].unsqueeze(1),
                    "mask": mask_flat[sb:sb + sub_bs].unsqueeze(1),
                })
                sb_bids = bid_flat[sb:sb + sub_bs]
                sb_w = w_flat[sb:sb + sub_bs]
                for axis in ("CE", "CU", "PC", "PQ"):
                    num_acc[axis].scatter_add_(0, sb_bids, sb_preds[axis] * sb_w)
                    denom_acc[axis].scatter_add_(0, sb_bids, sb_w)

            # Inverse Normalize on GPU: x*std + mean (both Python floats).
            return {
                axis: ((num_acc[axis] / denom_acc[axis]) * tr[axis].std + tr[axis].mean
                       ).to(torch.float32)
                for axis in ("CE", "CU", "PC", "PQ")
            }

    def _compute_batch(
        self,
        batch,
        new_indices: list,
        T: int,
        sr: int,
        eff_lengths: torch.Tensor,
        new_cuts: list,
        new_cut_ids: list,
        eff_secs_list: list,
        audio_secs: float,
        _rf,
    ) -> Optional[PendingBatch]:
        try:
            with _rf("mos/h2d_transfer"):
                with torch.cuda.stream(self._transfer_stream):
                    # Slice on the pinned tensor and gather on GPU — fancy
                    # indexing on CPU would force a sync H2D and break overlap.
                    audio_gpu = batch.audio[:, :T].to(
                        self._device, non_blocking=True,
                    )
                    if len(new_indices) != batch.audio.shape[0]:
                        idx = torch.tensor(
                            new_indices, dtype=torch.long, device=self._device,
                        )
                        audio_gpu = audio_gpu.index_select(0, idx)

            default_stream = torch.cuda.current_stream(self._device)
            default_stream.wait_stream(self._transfer_stream)

            utmos_out = squim_out = dns_out = None
            aes_means: Optional[dict] = None

            with _rf("mos/gpu_compute"):
                if self._use_dnsmos:
                    s_dns = self._dnsmos_stream
                    s_dns.wait_stream(default_stream)
                    audio_gpu.record_stream(s_dns)
                    with torch.cuda.stream(s_dns):
                        with _rf("mos/dnsmos_stft"):
                            spec = self._dnsmos_spec(audio_gpu)
                        with _rf("mos/dnsmos_forward"):
                            dns_out = self._dnsmos(spec)[:, 0]

                if self._use_utmos:
                    with _rf("mos/utmos_forward"):
                        utmos_out = self._utmos(audio_gpu, sr)

                if self._use_squim:
                    with _rf("mos/squim_forward"):
                        squim_out = self._squim(audio_gpu)

                if self._use_audiobox:
                    aes_means = self._audiobox_compute(
                        audio_gpu, T, sr, eff_lengths, _rf,
                    )

                if self._dnsmos_stream is not None:
                    default_stream.wait_stream(self._dnsmos_stream)

            with _rf("mos/d2h_post"):
                utmos_t = (
                    utmos_out.to("cpu", dtype=torch.float32, non_blocking=True)
                    if utmos_out is not None else None
                )
                stoi_t = pesq_t = si_sdr_t = dns_t = None
                if squim_out is not None:
                    sq_stoi, sq_pesq, sq_si_sdr = squim_out
                    stoi_t = sq_stoi.to("cpu", dtype=torch.float32, non_blocking=True)
                    pesq_t = sq_pesq.to("cpu", dtype=torch.float32, non_blocking=True)
                    si_sdr_t = sq_si_sdr.to("cpu", dtype=torch.float32, non_blocking=True)
                if dns_out is not None:
                    dns_t = dns_out.to("cpu", dtype=torch.float32, non_blocking=True)

                aes_CE_t = aes_CU_t = aes_PC_t = aes_PQ_t = None
                if aes_means is not None:
                    aes_CE_t = aes_means["CE"].to("cpu", non_blocking=True)
                    aes_CU_t = aes_means["CU"].to("cpu", non_blocking=True)
                    aes_PC_t = aes_means["PC"].to("cpu", non_blocking=True)
                    aes_PQ_t = aes_means["PQ"].to("cpu", non_blocking=True)

                event = torch.cuda.Event()
                event.record()

            return PendingBatch(
                event=event,
                cut_ids=new_cut_ids, cuts=new_cuts, audio_secs=audio_secs,
                eff_secs=eff_secs_list,
                utmos_t=utmos_t, stoi_t=stoi_t, pesq_t=pesq_t,
                si_sdr_t=si_sdr_t, dns_t=dns_t,
                aes_CE_t=aes_CE_t, aes_CU_t=aes_CU_t,
                aes_PC_t=aes_PC_t, aes_PQ_t=aes_PQ_t,
            )

        except torch.cuda.OutOfMemoryError:
            if self._fail_fast_oom:
                logger.error("CUDA OOM — fail_fast_oom set, aborting.")
                raise
            logger.exception("CUDA OOM (cut_ids: %s…), skipping batch.", new_cut_ids[:3])
            return None
        except Exception:
            logger.exception("Inference failed (cut_ids: %s…), skipping batch.", new_cut_ids[:3])
            return None

    def run(self, output_dir: str) -> dict:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_jsonl = Path(output_dir) / f"part_{self._rank:04d}.jsonl"

        seen_ids = load_seen_cut_ids(output_jsonl)
        metrics = self._metrics

        profiler = make_profiler(self._cfg, self._rank, output_dir)
        prof_ctx = profiler if profiler is not None else contextlib.nullcontext()

        def _rf(name: str) -> StageTimer:
            return StageTimer(
                metrics, name,
                profiler_rf=(torch.profiler.record_function(name)
                             if profiler is not None else None),
            )

        # Drained at the END of the next iteration so the previous batch's
        # d2h DMA overlaps with the next batch's H2D + model launches.
        pending: Optional[PendingBatch] = None

        with open(output_jsonl, mode="a") as out_f:

            def _flush(pb: PendingBatch) -> tuple:
                with _rf("mos/d2h_sync"):
                    pb.event.synchronize()

                utmos_cpu = pb.utmos_t.tolist() if pb.utmos_t is not None else []
                stoi_cpu = pb.stoi_t.tolist() if pb.stoi_t is not None else []
                pesq_cpu = pb.pesq_t.tolist() if pb.pesq_t is not None else []
                si_sdr_cpu = pb.si_sdr_t.tolist() if pb.si_sdr_t is not None else []
                dns_cpu = pb.dns_t.tolist() if pb.dns_t is not None else []
                ce_cpu = pb.aes_CE_t.tolist() if pb.aes_CE_t is not None else []
                cu_cpu = pb.aes_CU_t.tolist() if pb.aes_CU_t is not None else []
                pc_cpu = pb.aes_PC_t.tolist() if pb.aes_PC_t is not None else []
                pq_cpu = pb.aes_PQ_t.tolist() if pb.aes_PQ_t is not None else []

                dnsmos_key = f"dnsmos_{self._dnsmos_variant}"

                with _rf("mos/jsonl_write"):
                    for j, (cut_id, cut, eff_s) in enumerate(
                        zip(pb.cut_ids, pb.cuts, pb.eff_secs)
                    ):
                        eval_secs = round(eff_s, 3)
                        per_metric: dict = {}
                        if self._use_utmos:
                            per_metric["utmos"] = {
                                "eval_secs": eval_secs,
                                "score": {"utmos": round(utmos_cpu[j], 4)},
                            }
                        if self._use_squim:
                            per_metric["squim"] = {
                                "eval_secs": eval_secs,
                                "score": {
                                    "stoi": round(stoi_cpu[j], 4),
                                    "pesq": round(pesq_cpu[j], 4),
                                    "si_sdr": round(si_sdr_cpu[j], 4),
                                },
                            }
                        if self._use_dnsmos:
                            per_metric[dnsmos_key] = {
                                "eval_secs": eval_secs,
                                "score": {self._dnsmos_variant: round(dns_cpu[j], 4)},
                            }
                        if self._use_audiobox:
                            per_metric["audiobox"] = {
                                "eval_secs": eval_secs,
                                "score": {
                                    "CE": round(ce_cpu[j], 4),
                                    "CU": round(cu_cpu[j], 4),
                                    "PC": round(pc_cpu[j], 4),
                                    "PQ": round(pq_cpu[j], 4),
                                },
                            }
                        record = {
                            "cut_id": cut_id,
                            "duration": round(cut.duration, 3),
                            "metrics": per_metric,
                        }
                        # Explicit dataset tag (unified ids): lets the dedup
                        # quality join key on (dataset, cut_id) robustly instead
                        # of a path heuristic.  Set to retention's manifest tag.
                        if self._cfg.get("dataset"):
                            record["dataset"] = self._cfg["dataset"]
                        out_f.write(json.dumps(record, separators=(",", ":")))
                        out_f.write("\n")
                        seen_ids.add(cut_id)

                return len(pb.cut_ids), pb.audio_secs

            def _drain(pb: Optional[PendingBatch]) -> None:
                if pb is None:
                    return
                try:
                    n, s = _flush(pb)
                    metrics.add_batch(n, s, status="ok")
                    if metrics.batches % 50 == 0:
                        out_f.flush()
                    metrics.maybe_periodic_log(logger, every_n=50)
                except Exception:
                    logger.exception("Flush failed for pending batch.")
                finally:
                    if profiler is not None:
                        profiler.step()

            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=self._dtype), prof_ctx:
                _loader_iter = iter(self._loader)
                while not self._stop_requested:
                    try:
                        batch = next(_loader_iter)
                    except StopIteration:
                        break
                    except Exception:
                        logger.exception("Data loader error — draining and stopping.")
                        break

                    new_indices = [
                        i for i, cid in enumerate(batch.cut_ids)
                        if cid not in seen_ids
                    ]
                    if not new_indices:
                        _drain(pending)
                        pending = None
                        metrics.add_batch(len(batch.cut_ids), status="skipped")
                        if profiler is not None:
                            profiler.step()
                        continue

                    sr = batch.sr
                    max_samples = int(self._truncate_secs * sr)
                    T = min(max_samples, batch.audio.shape[1])
                    eff_lengths = batch.lengths[new_indices].clamp(max=max_samples)
                    eff_secs_list = (eff_lengths.float() / sr).tolist()
                    audio_secs = sum(eff_secs_list)
                    new_cuts = [batch.cuts[i] for i in new_indices]
                    new_cut_ids = [batch.cut_ids[i] for i in new_indices]

                    current_pending = self._compute_batch(
                        batch, new_indices, T, sr, eff_lengths,
                        new_cuts, new_cut_ids, eff_secs_list, audio_secs, _rf,
                    )

                    _drain(pending)
                    pending = current_pending

                    if current_pending is None:
                        metrics.add_batch(len(new_indices), status="failed")
                        if profiler is not None:
                            profiler.step()

                _drain(pending)
                pending = None

            out_f.flush()

        if profiler is not None:
            logger.info(
                "Profiler key averages (sort: cuda_time_total):\n%s",
                profiler.key_averages().table(sort_by="cuda_time_total", row_limit=20),
            )

        metrics.final_summary(logger)
        stats = metrics.as_stats()
        stats["stop_requested"] = self._stop_requested
        return stats
