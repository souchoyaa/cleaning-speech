#!/usr/bin/env python3
"""Build a dedup *evaluation* dataset from a prepared Lhotse Shar.

Keeps every original cut and adds ``--extra-frac`` (default 0.30) synthetic
samples split by a fixed proportion table of duplicate/near-duplicate/negative
types.  Every output cut has a unique id; synthetic cuts carry their provenance
(which original they derive from + the duplicate type + ground-truth match
expectations) inside ``cut.custom["dedup_eval"]``.  A sidecar
``ground_truth.jsonl`` mirrors that for convenience.

Proportions of the synthetic block (per the eval spec):
    exact                 10%   verbatim copy                (audio dup, text dup)
    text_slight_change    15%   same audio, 1-2 words edited  (audio dup, text near)
    degraded_quality      30%   realistic transcodes/SR/noise (audio near-dup, text dup)
    same_text_diff_speaker15%   pitch/tempo-shifted "voice"   (text dup, NOT audio dup)
    partial_crop          20%   contiguous 40-70% audio crop  (audio partial, text dup)
    hard_negative         10%   voice-shift + heavy text edit (NOT a dup of anything)

"degraded_quality" models cross-dataset re-uploads: MP3/Opus/Vorbis re-encode,
mu-law (telephony), sample-rate round-trips, and multi-hop chains, plus light
background noise / re-normalization — same audio, not bit-exact.

MOS-quality block (optional, ``--mos-count N`` > 0): a separate set of cuts that
exercises the *quality* pipeline (B.2).  Each is a degraded copy of a distinct
clean original (its quality reference), drawn from a degradation family at a
sampled severity, so per-metric AUROC can be measured against a known label and
plotted vs severity.  Families and the axis each is meant to stress:
    clean_control  pristine (FLAC round-trip only)            — positive control
    awgn           synthetic white noise, SNR sweep           — dnsmos (sim ref)
    real_noise     MUSAN noise, SNR sweep                     — dnsmos (real)
    music          MUSAN music mixed in, SNR sweep            — audiobox PC / CU
    reverb         real room impulse response (RIRS_NOISES)   — utmos (profile)
    crosstalk      a second speaker mixed in, SIR sweep       — stoi / si_sdr
    clipping       hard clipping, threshold sweep             — dnsmos
    telephony      8 kHz band-limit + mu-law                  — si_sdr / dnsmos
    naturalness    Griffin-Lim resynthesis (clean SNR!)       — utmos (unique)
Ground truth for these rows carries a nested ``mos_eval`` dict (family, variant,
severity, ``expect_low_quality``, ``quality_tier`` 0-3, ``primary_axis``).
real_noise / music / reverb need ``--musan-dir`` / ``--rir-dir``; if a pool is
missing the family falls back to its synthetic analogue (awgn) and is flagged.

Standalone low-quality block (``--standalone-lowq-count N``): a severe degradation
PLUS a heavily-altered transcript, so the cut has NO twin (unique at text AND
audio level) and is labelled low quality (``mos_eval.standalone=true``).  These
are the cuts that exercise the unique-sample gate_drop path in retention.

Source audio is read read-only via the Shar tar offsets.  Output is a fresh
Shar (worker_XX/ + merged shar_index.json).  Reproducible: every op's randomness
is seeded from (--seed, out_id).

Run inside the container (see RESULTS.md).  A full "v3" build with every block:
  python3 .../dup_retrieval/eval/make_eval_dataset.py \
    --src-dirs <librispeech train_clean_100> <train_clean_360> \
    --out-dir  <eval_shar_out>/en --extra-frac 0.30 --num-workers 32 --seed 1 \
    --mos-count 18000 --standalone-lowq-count 4000 \
    --musan-dir <assets>/musan --rir-dir <assets>/RIRS_NOISES/simulated_rirs
"""

import argparse
import gzip
import hashlib
import io
import json
import multiprocessing as mp
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly, fftconvolve
from lhotse import MonoCut, Recording, SupervisionSegment
from lhotse.shar import SharWriter

AUDIO_EXT = "flac"

PROPORTIONS = [
    ("exact", 0.10),
    ("text_slight_change", 0.15),
    ("degraded_quality", 0.30),
    ("same_text_diff_speaker", 0.15),
    ("partial_crop", 0.20),
    ("hard_negative", 0.10),
]
FILLERS = ["the", "and", "a", "to", "of", "in", "is", "was", "that", "with"]


def _rng(*parts):
    h = hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()
    return np.random.RandomState(int(h[:8], 16))


def _new_id(*parts):
    return hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()[:24]


# ---------------------------------------------------------------------------
# Source reader (read-only, random-access)
# ---------------------------------------------------------------------------

def read_source_index(src_dir: Path):
    """Return list of dicts {id, text, custom, tar_path, offset, size, duration}."""
    idx = json.loads((src_dir / "shar_index.json").read_text())
    cuts_rel, rec_rel = idx["fields"]["cuts"], idx["fields"]["recording"]

    def shard_num(name):
        for tok in Path(name).name.split("."):
            if tok.isdigit():
                return tok
        return None

    rec_by_key = {(str(Path(r).parent), shard_num(r)): r for r in rec_rel}
    rows = []
    for cr in cuts_rel:
        rr = rec_by_key[(str(Path(cr).parent), shard_num(cr))]
        tar_path = str(src_dir / rr)
        offsets = {}
        with tarfile.open(tar_path, "r:") as tf:
            for ti in tf:
                if ti.isfile():
                    base, _, ext = ti.name.rpartition(".")
                    if ext == AUDIO_EXT:
                        offsets[base] = (ti.offset_data, ti.size)
        with gzip.open(src_dir / cr, "rt") as f:
            for line in f:
                cd = json.loads(line)
                cid = cd["id"]
                if cid not in offsets:
                    continue
                sup = (cd.get("supervisions") or [{}])[0]
                off, size = offsets[cid]
                rows.append({
                    "id": cid,
                    "text": (sup.get("text") or "").strip(),
                    "custom": cd.get("custom") or {},
                    "tar_path": tar_path, "offset": off, "size": size,
                    "duration": float(cd.get("duration") or 0.0),
                })
    return rows


def read_audio(tar_path, offset, size):
    with open(tar_path, "rb") as fh:
        fh.seek(offset)
        raw = fh.read(size)
    y, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    return y, sr


# ---------------------------------------------------------------------------
# Audio ops (worker side) — realistic degradations
# ---------------------------------------------------------------------------

def _codec_roundtrip(y, sr, codec):
    fmt_sub = {"mp3": ("MP3", None), "ogg_vorbis": ("OGG", "VORBIS"),
               "opus": ("OGG", "OPUS"), "mulaw": ("WAV", "ULAW")}[codec]
    buf = io.BytesIO()
    if fmt_sub[1]:
        sf.write(buf, y, sr, format=fmt_sub[0], subtype=fmt_sub[1])
    else:
        sf.write(buf, y, sr, format=fmt_sub[0])
    buf.seek(0)
    yr, _ = sf.read(buf, dtype="float32", always_2d=False)
    if yr.ndim > 1:
        yr = yr.mean(axis=1)
    # codecs may pad/trim a few samples; clip back to source length.
    if len(yr) >= len(y):
        return yr[:len(y)]
    return np.pad(yr, (0, len(y) - len(yr)))


def _resample_rt(y, sr, mid_sr):
    from math import gcd
    g1 = gcd(mid_sr, sr)
    down = resample_poly(y, mid_sr // g1, sr // g1)         # sr -> mid_sr
    g2 = gcd(sr, mid_sr)
    up = resample_poly(down, sr // g2, mid_sr // g2)        # mid_sr -> sr
    if len(up) >= len(y):
        return up[:len(y)].astype(np.float32)
    return np.pad(up, (0, len(y) - len(up))).astype(np.float32)


def _bg_noise(y, snr_db, rng):
    p = float(np.mean(y ** 2)) + 1e-12
    n = rng.normal(0, np.sqrt(p / (10 ** (snr_db / 10))), size=y.shape).astype(np.float32)
    return np.clip(y + n, -1.0, 1.0)


def _volume(y, gain_db):
    return np.clip(y * (10 ** (gain_db / 20.0)), -1.0, 1.0).astype(np.float32)


def _voice_shift(y, sr, factor):
    """Pitch+tempo shift via resample (proxy for a different speaker)."""
    from math import gcd
    n_out = max(1, int(round(len(y) / factor)))
    g = gcd(n_out, len(y))
    return resample_poly(y, n_out // g, len(y) // g).astype(np.float32)


def _crop(y, start_frac, len_frac):
    n = len(y)
    s = int(start_frac * n)
    e = min(n, s + max(1, int(len_frac * n)))
    return y[s:e]


# ----- MOS-quality ops (real/realistic degradations) -----------------------

def _load_asset(path, sr):
    """Read an external audio file (noise/music/RIR), mono, resampled to *sr*."""
    a, asr = sf.read(path, dtype="float32", always_2d=False)
    if a.ndim > 1:
        a = a.mean(axis=1)
    if asr != sr and len(a):
        from math import gcd
        g = gcd(int(sr), int(asr))
        a = resample_poly(a, int(sr) // g, int(asr) // g).astype(np.float32)
    return np.ascontiguousarray(a, dtype=np.float32)


def _fit_len(x, n, rng):
    """Tile or random-crop background *x* to exactly *n* samples."""
    if len(x) == 0:
        return np.zeros(n, np.float32)
    if len(x) < n:
        x = np.tile(x, int(np.ceil(n / len(x))))
    s = int(rng.randint(0, max(1, len(x) - n + 1)))
    return x[s:s + n].astype(np.float32)


def _mix_at_snr(y, bg, snr_db):
    """Add background *bg* to *y* scaled to the requested SNR (dB)."""
    py = float(np.mean(y ** 2)) + 1e-12
    pb = float(np.mean(bg ** 2)) + 1e-12
    g = float(np.sqrt(py / (pb * (10 ** (snr_db / 10.0)))))
    return np.clip(y + g * bg, -1.0, 1.0).astype(np.float32)


def _real_bg(y, sr, files, snr_db, rng):
    """Mix a random real noise/music clip at *snr_db*.  Falls back to AWGN."""
    if not files:
        return _bg_noise(y, snr_db, rng)
    bg = _fit_len(_load_asset(files[rng.randint(len(files))], sr), len(y), rng)
    return _mix_at_snr(y, bg, snr_db)


def _reverb(y, sr, rir_files, rng):
    """Convolve with a real room impulse response, preserve length + peak."""
    if not rir_files:
        return y
    h = _load_asset(rir_files[rng.randint(len(rir_files))], sr)
    if len(h) == 0:
        return y
    h = h / (np.max(np.abs(h)) + 1e-9)
    wet = fftconvolve(y, h)[:len(y)]
    pk_in, pk_out = np.max(np.abs(y)) + 1e-9, np.max(np.abs(wet)) + 1e-9
    return (wet * pk_in / pk_out).astype(np.float32)


def _crosstalk(y, sr, interferers, sir_db, rng):
    """Mix a second speaker (another source cut) at signal-to-interferer ratio."""
    if not interferers:
        return y
    tp, off, size = interferers[rng.randint(len(interferers))]
    o, osr = read_audio(tp, off, size)
    if osr != sr and len(o):
        from math import gcd
        g = gcd(int(sr), int(osr))
        o = resample_poly(o, int(sr) // g, int(osr) // g).astype(np.float32)
    return _mix_at_snr(y, _fit_len(o, len(y), rng), sir_db)


def _clip_distort(y, threshold):
    """Hard-clip at *threshold* * peak, then renormalize to the original peak."""
    pk = np.max(np.abs(y)) + 1e-9
    t = threshold * pk
    return (np.clip(y, -t, t) / t * pk).astype(np.float32)


def _griffin_lim(y, sr, n_iter):
    """Magnitude-STFT resynthesis: near-clean SNR but unnatural (vocoder-like)."""
    import librosa
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))
    yr = librosa.griffinlim(S, n_iter=int(n_iter), hop_length=256)
    yr = yr[:len(y)] if len(yr) >= len(y) else np.pad(yr, (0, len(y) - len(yr)))
    pk_in, pk_out = np.max(np.abs(y)) + 1e-9, np.max(np.abs(yr)) + 1e-9
    return (yr * pk_in / pk_out).astype(np.float32)


def apply_ops(y, sr, ops, rng, assets=None):
    assets = assets or {}
    for op in ops:
        name, p = op["name"], op.get("params", {})
        if name == "codec":
            y = _codec_roundtrip(y, sr, p["codec"])
        elif name == "resample_rt":
            y = _resample_rt(y, sr, p["mid_sr"])
        elif name == "bg_noise":
            y = _bg_noise(y, p["snr_db"], rng)
        elif name == "volume":
            y = _volume(y, p["gain_db"])
        elif name == "voice_shift":
            y = _voice_shift(y, sr, p["factor"])
        elif name == "crop":
            y = _crop(y, p["start_frac"], p["len_frac"])
        elif name == "real_noise":
            y = _real_bg(y, sr, assets.get("noise_files", []), p["snr_db"], rng)
        elif name == "music":
            y = _real_bg(y, sr, assets.get("music_files", []), p["snr_db"], rng)
        elif name == "reverb":
            y = _reverb(y, sr, assets.get("rir_files", []), rng)
        elif name == "crosstalk":
            y = _crosstalk(y, sr, assets.get("interferers", []), p["sir_db"], rng)
        elif name == "clip_distort":
            y = _clip_distort(y, p["threshold"])
        elif name == "griffin_lim":
            y = _griffin_lim(y, sr, p["n_iter"])
        else:
            raise ValueError(name)
    return y.astype(np.float32)


# ---------------------------------------------------------------------------
# Text edits
# ---------------------------------------------------------------------------

def edit_text(text, frac, rng):
    """Replace ~frac of the words with fillers (frac small = slight, large = heavy)."""
    words = text.split()
    if len(words) < 3:
        return text + " " + str(rng.choice(FILLERS)), 1
    k = max(1, int(round(frac * len(words))))
    idxs = rng.choice(len(words), size=min(k, len(words)), replace=False)
    for i in idxs:
        words[i] = str(rng.choice(FILLERS))
    return " ".join(words), int(len(idxs))




# ---------------------------------------------------------------------------
# Degradation plan (main side, deterministic)
# ---------------------------------------------------------------------------

def plan_degradation(rng):
    """Return (ops, label) for one realistic 'degraded_quality' sample."""
    choice = rng.choice([
        "mp3", "ogg_vorbis", "opus", "mulaw",
        "resample8", "resample11", "reupload_chain", "bg_noise", "volume",
    ], p=[0.16, 0.16, 0.12, 0.08, 0.08, 0.08, 0.16, 0.08, 0.08])
    if choice in ("mp3", "ogg_vorbis", "opus", "mulaw"):
        return [{"name": "codec", "params": {"codec": choice}}], f"codec:{choice}"
    if choice == "resample8":           # telephone-band round-trip
        return [{"name": "resample_rt", "params": {"mid_sr": 8000}}], "resample:8k"
    if choice == "resample11":
        return [{"name": "resample_rt", "params": {"mid_sr": 11025}}], "resample:11k"
    if choice == "reupload_chain":      # SR round-trip then lossy re-encode
        mid = int(rng.choice([8000, 11025]))
        codec = str(rng.choice(["mp3", "ogg_vorbis", "opus"]))
        return ([{"name": "resample_rt", "params": {"mid_sr": mid}},
                 {"name": "codec", "params": {"codec": codec}}],
                f"chain:resample{mid//1000}k+{codec}")
    if choice == "bg_noise":
        snr = float(rng.choice([20.0, 25.0, 30.0, 35.0]))
        return [{"name": "bg_noise", "params": {"snr_db": snr}}], f"noise:{snr}dB"
    gain = float(rng.choice([-9.0, -6.0, 6.0, 9.0]))
    return [{"name": "volume", "params": {"gain_db": gain}}], f"volume:{gain}dB"


# ---------------------------------------------------------------------------
# MOS-quality block (exercises the quality pipeline B.2)
# ---------------------------------------------------------------------------

# (family, share).  Shares are renormalized over the families actually kept.
MOS_PROPORTIONS = [
    ("clean_control", 0.06),
    ("awgn",          0.12),
    ("real_noise",    0.16),
    ("music",         0.12),
    ("reverb",        0.12),
    ("crosstalk",     0.10),
    ("clipping",      0.10),
    ("telephony",     0.10),
    ("naturalness",   0.12),
]

# Coarse SNR/SIR -> quality tier (0 worst .. 3 clean) and low-quality cutoff.
_SNR_SWEEP = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0]


def _snr_tier(snr):
    return 0 if snr <= 5 else 1 if snr <= 15 else 2 if snr <= 25 else 3


def plan_mos(family, rng):
    """Return (ops, mos_eval) for one MOS-quality sample of *family*.

    ``mos_eval`` records the ground-truth label set: the degradation variant, its
    severity, the expected quality tier (0-3), whether we expect a curator to
    treat it as low quality, and the metric that *should* catch it (primary_axis).
    """
    ev = {"family": family, "fallback": False, "expect_clean_snr": False,
          "expect_perceptual_penalty": False}

    if family == "clean_control":
        ev.update(variant="passthrough", severity=None, expect_low_quality=False,
                  quality_tier=3, primary_axis=None)
        return [], ev

    if family in ("awgn", "real_noise"):
        snr = float(rng.choice(_SNR_SWEEP))
        op = "bg_noise" if family == "awgn" else "real_noise"
        ev.update(variant=f"snr{snr:g}", severity=snr,
                  expect_low_quality=bool(snr <= 15), quality_tier=_snr_tier(snr),
                  primary_axis="dnsmos")
        return [{"name": op, "params": {"snr_db": snr}}], ev

    if family == "music":
        snr = float(rng.choice([0.0, 5.0, 10.0, 15.0, 20.0]))
        ev.update(variant=f"music_snr{snr:g}", severity=snr,
                  expect_low_quality=bool(snr <= 10), quality_tier=_snr_tier(snr),
                  primary_axis="audiobox_PC")
        return [{"name": "music", "params": {"snr_db": snr}}], ev

    if family == "reverb":
        # Real RIR.  We do not know RT60 per file, so reverb is labelled "fine"
        # (matches the stance that reverberant real-room speech is legitimate)
        # but flagged: every perceptual MOS metric still penalizes it — that gap
        # is the measurement (a hard MOS gate would over-filter real rooms).
        ev.update(variant="real_rir", severity=None, expect_low_quality=False,
                  quality_tier=2, primary_axis="utmos", expect_perceptual_penalty=True)
        return [{"name": "reverb", "params": {}}], ev

    if family == "crosstalk":
        sir = float(rng.choice([0.0, 5.0, 10.0, 20.0]))
        ev.update(variant=f"sir{sir:g}", severity=sir,
                  expect_low_quality=bool(sir <= 5), quality_tier=_snr_tier(sir),
                  primary_axis="stoi")
        return [{"name": "crosstalk", "params": {"sir_db": sir}}], ev

    if family == "clipping":
        thr = float(rng.choice([0.1, 0.2, 0.3, 0.5]))
        tier = 0 if thr <= 0.1 else 1 if thr <= 0.2 else 2 if thr <= 0.3 else 3
        ev.update(variant=f"clip{thr:g}", severity=thr,
                  expect_low_quality=bool(thr <= 0.2), quality_tier=tier,
                  primary_axis="dnsmos")
        return [{"name": "clip_distort", "params": {"threshold": thr}}], ev

    if family == "telephony":
        ops = [{"name": "resample_rt", "params": {"mid_sr": 8000}}]
        if rng.rand() < 0.5:
            ops.append({"name": "codec", "params": {"codec": "mulaw"}})
        ev.update(variant="band8k" + ("+mulaw" if len(ops) > 1 else ""),
                  severity=8000, expect_low_quality=True, quality_tier=1,
                  primary_axis="si_sdr")
        return ops, ev

    if family == "naturalness":
        n_iter = int(rng.choice([1, 2, 4, 8, 16, 32]))
        tier = 0 if n_iter <= 2 else 1 if n_iter <= 8 else 2
        ev.update(variant=f"gl{n_iter}", severity=n_iter, expect_low_quality=True,
                  quality_tier=tier, primary_axis="utmos", expect_clean_snr=True)
        return [{"name": "griffin_lim", "params": {"n_iter": n_iter}}], ev

    raise ValueError(family)


# ---------------------------------------------------------------------------
# Standalone low-quality (degrade-in-place: no clean twin, altered transcript)
# ---------------------------------------------------------------------------

def plan_standalone(rng):
    """A genuinely standalone bad recording: a severe degradation, paired with a
    heavily-altered transcript so the cut is unique at BOTH the text and audio
    level (no twin) — the case needed to validate the unique-sample gate_drop."""
    choice = str(rng.choice(["real_noise", "clip", "telephony", "music"]))
    ev = {"family": choice, "standalone": True, "fallback": False,
          "expect_clean_snr": False, "expect_perceptual_penalty": False,
          "expect_low_quality": True, "quality_tier": 0}
    if choice == "real_noise":
        snr = float(rng.choice([0.0, 5.0]))
        ev.update(variant=f"snr{snr:g}", severity=snr, primary_axis="dnsmos")
        return [{"name": "real_noise", "params": {"snr_db": snr}}], ev
    if choice == "music":
        snr = float(rng.choice([0.0, 5.0]))
        ev.update(variant=f"music_snr{snr:g}", severity=snr, primary_axis="audiobox_PC")
        return [{"name": "music", "params": {"snr_db": snr}}], ev
    if choice == "clip":
        thr = float(rng.choice([0.1, 0.15]))
        ev.update(variant=f"clip{thr:g}", severity=thr, primary_axis="dnsmos")
        return [{"name": "clip_distort", "params": {"threshold": thr}}], ev
    ev.update(variant="band8k+mulaw", severity=8000, quality_tier=1, primary_axis="si_sdr")
    return ([{"name": "resample_rt", "params": {"mid_sr": 8000}},
             {"name": "codec", "params": {"codec": "mulaw"}}], ev)


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _worker(args):
    wid, items, out_dir, assets, seed = args
    wd = Path(out_dir) / f"worker_{wid:02d}"
    wd.mkdir(parents=True, exist_ok=True)
    gt_rows = []
    with SharWriter(output_dir=str(wd), fields={"recording": AUDIO_EXT},
                    shard_size=10_000_000) as w:
        for it in items:
            y, sr = read_audio(it["audio"]["tar_path"], it["audio"]["offset"],
                               it["audio"]["size"])
            rng = _rng(seed, "op", it["out_id"])
            y = apply_ops(y, sr, it["ops"], rng, assets)
            buf = io.BytesIO()
            sf.write(buf, y, sr, format="FLAC")
            rec = Recording.from_bytes(buf.getvalue(), recording_id=it["out_id"])
            cut = MonoCut(
                id=it["out_id"], start=0.0, duration=rec.duration, channel=0,
                recording=rec,
                supervisions=[SupervisionSegment(
                    id=it["out_id"], recording_id=it["out_id"], start=0.0,
                    duration=rec.duration, text=it["text"],
                    language=it.get("language", "en"))],
                custom=it["custom"],
            )
            w.write(cut)
            row = {"cut_id": it["out_id"], "duration": float(rec.duration)}
            row.update(it["custom"]["dedup_eval"])
            if "mos_eval" in it["custom"]:
                row["mos_eval"] = it["custom"]["mos_eval"]
            gt_rows.append(row)
    return wid, gt_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src-dirs", required=True, nargs="+", type=Path,
                    help="One or more prepared Shar dirs; combined into the base set.")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--extra-frac", type=float, default=0.30)
    ap.add_argument("--max-base", type=int, default=0,
                    help="If >0, cap the base set to this many cuts (for smoke tests).")
    ap.add_argument("--num-workers", type=int, default=32)
    ap.add_argument("--seed", type=int, default=1)
    # ----- MOS-quality block -----
    ap.add_argument("--mos-count", type=int, default=0,
                    help="If >0, add this many labelled quality cuts (MOS block).")
    ap.add_argument("--musan-dir", type=Path, default=None,
                    help="MUSAN root (uses noise/ and music/) for real-noise/music families.")
    ap.add_argument("--rir-dir", type=Path, default=None,
                    help="Directory of RIR wavs (recursed) for the reverb family.")
    ap.add_argument("--standalone-lowq-count", type=int, default=0,
                    help="If >0, add this many STANDALONE low-quality cuts (degraded "
                         "audio + altered transcript = no twin) to test the unique gate.")
    args = ap.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(args.seed)
    src = []
    for d in args.src_dirs:
        rows = read_source_index(d)
        print(f"  {d}: {len(rows)} source cuts")
        src.extend(rows)
    if args.max_base and len(src) > args.max_base:
        src = [src[i] for i in rng.permutation(len(src))[:args.max_base]]
        print(f"  capped base to {len(src)} (--max-base)")
    n_base = len(src)
    print(f"  total base: {n_base} source cuts")

    n_synth = int(round(args.extra_frac * n_base))
    # integer split across types (remainder -> degraded_quality)
    counts = {t: int(round(p * n_synth)) for t, p in PROPORTIONS}
    counts["degraded_quality"] += n_synth - sum(counts.values())
    print(f"  synthetic block: {n_synth} cuts -> {counts}")

    # one distinct source original per synthetic
    perm = rng.permutation(n_base)
    src_for_synth = [src[i] for i in perm[:n_synth]]

    def passthrough_custom(r):
        c = {k: r["custom"].get(k) for k in ("speaker_id", "chapter_id") if k in r["custom"]}
        return c

    items = []

    # 1) all originals (verbatim passthrough)
    for r in src:
        c = passthrough_custom(r)
        c["dedup_eval"] = {"is_synthetic": False, "dup_type": "original",
                           "original_id": None, "expect_audio_dup_of": None,
                           "expect_text_dup_of": None}
        items.append({"out_id": r["id"], "text": r["text"], "ops": [],
                      "audio": {"tar_path": r["tar_path"], "offset": r["offset"],
                                "size": r["size"]}, "custom": c})

    # 2) synthetics
    si = 0
    for dup_type, _ in PROPORTIONS:
        for _ in range(counts[dup_type]):
            A = src_for_synth[si]; si += 1
            out_id = _new_id(args.seed, dup_type, A["id"])
            prng = _rng(args.seed, "plan", out_id)
            ops, text, ev, params = [], A["text"], {}, {}

            if dup_type == "exact":
                ev = {"expect_audio_dup_of": A["id"], "expect_text_dup_of": A["id"]}
            elif dup_type == "text_slight_change":
                text, n = edit_text(A["text"], 0.12, prng)
                params = {"n_edits": n}
                ev = {"expect_audio_dup_of": A["id"], "expect_text_dup_of": A["id"]}
            elif dup_type == "degraded_quality":
                ops, label = plan_degradation(prng)
                params = {"degradation": label}
                ev = {"expect_audio_dup_of": A["id"], "expect_text_dup_of": A["id"]}
            elif dup_type == "same_text_diff_speaker":
                f = float(prng.choice([0.80, 0.85, 1.16, 1.22]))
                ops = [{"name": "voice_shift", "params": {"factor": f}}]
                params = {"voice_factor": f}
                ev = {"expect_audio_dup_of": None, "expect_text_dup_of": A["id"]}
            elif dup_type == "partial_crop":
                lf = float(prng.uniform(0.4, 0.7))
                sf_ = float(prng.uniform(0.0, 1.0 - lf))
                ops = [{"name": "crop", "params": {"start_frac": sf_, "len_frac": lf}}]
                params = {"crop_start_frac": round(sf_, 3), "crop_len_frac": round(lf, 3)}
                ev = {"expect_audio_dup_of": A["id"], "expect_text_dup_of": A["id"]}
            elif dup_type == "hard_negative":
                f = float(prng.choice([0.80, 0.85, 1.16, 1.22]))
                ops = [{"name": "voice_shift", "params": {"factor": f}}]
                text, n = edit_text(A["text"], 0.45, prng)
                params = {"voice_factor": f, "n_edits": n}
                ev = {"expect_audio_dup_of": None, "expect_text_dup_of": None}

            c = passthrough_custom(A)
            c["dedup_eval"] = {"is_synthetic": True, "dup_type": dup_type,
                               "original_id": A["id"], **ev, "params": params}
            items.append({"out_id": out_id, "text": text, "ops": ops,
                          "audio": {"tar_path": A["tar_path"], "offset": A["offset"],
                                    "size": A["size"]}, "custom": c})

    # ----- Shared real-degradation assets (MOS + standalone blocks) -----
    mos_counts = {}
    n_standalone = 0
    assets = {}
    noise_files = music_files = rir_files = []
    if args.mos_count > 0 or args.standalone_lowq_count > 0:
        def _glob_audio(d, cap, seed):
            if not d or not Path(d).is_dir():
                return []
            files = sorted(str(p) for p in Path(d).rglob("*")
                           if p.suffix.lower() in (".wav", ".flac"))
            if cap and len(files) > cap:
                idx = sorted(np.random.RandomState(seed).permutation(len(files))[:cap])
                files = [files[i] for i in idx]
            return files

        noise_files = _glob_audio(args.musan_dir and args.musan_dir / "noise", 3000, args.seed + 1)
        music_files = _glob_audio(args.musan_dir and args.musan_dir / "music", 3000, args.seed + 2)
        rir_files = _glob_audio(args.rir_dir, 5000, args.seed + 3)
        iperm = np.random.RandomState(args.seed + 4).permutation(n_base)[:512]
        interferers = [(src[i]["tar_path"], src[i]["offset"], src[i]["size"]) for i in iperm]
        assets = {"noise_files": noise_files, "music_files": music_files,
                  "rir_files": rir_files, "interferers": interferers}
        print(f"  assets: noise={len(noise_files)} music={len(music_files)} "
              f"rir={len(rir_files)} interferers={len(interferers)}")

    # 3) MOS-quality block — labelled quality cuts (exercises pipeline B.2).
    if args.mos_count > 0:
        active = []
        for fam, share in MOS_PROPORTIONS:
            if fam == "reverb" and not rir_files:
                print("  [mos] no RIRs -> dropping 'reverb' family"); continue
            if fam == "music" and not music_files:
                print("  [mos] no music -> dropping 'music' family"); continue
            active.append((fam, share))
        tot = sum(s for _, s in active)
        mos_counts = {f: int(round(s / tot * args.mos_count)) for f, s in active}
        if mos_counts:
            big = max(mos_counts, key=mos_counts.get)
            mos_counts[big] += args.mos_count - sum(mos_counts.values())
        print(f"  MOS block: {args.mos_count} cuts -> {mos_counts}")

        mperm = np.random.RandomState(args.seed + 5).permutation(n_base)
        mi = 0
        for fam, _ in active:
            for _ in range(mos_counts[fam]):
                A = src[mperm[mi % n_base]]; mi += 1
                out_id = _new_id(args.seed, "mos", fam, A["id"], mi)
                prng = _rng(args.seed, "mosplan", out_id)
                ops, mev = plan_mos(fam, prng)
                if fam == "real_noise" and not noise_files:
                    mev["fallback"] = True
                mev["clean_ref_id"] = A["id"]
                # Truthful dedup labels: identical transcript (text dup of A);
                # audio is a near-dup of A except crosstalk (2nd speaker mixed in).
                audio_dup = None if fam == "crosstalk" else A["id"]
                c = passthrough_custom(A)
                c["dedup_eval"] = {"is_synthetic": True, "dup_type": f"mos:{fam}",
                                   "original_id": A["id"],
                                   "expect_audio_dup_of": audio_dup,
                                   "expect_text_dup_of": A["id"],
                                   "params": {"mos_family": fam}}
                c["mos_eval"] = mev
                items.append({"out_id": out_id, "text": A["text"], "ops": ops,
                              "audio": {"tar_path": A["tar_path"], "offset": A["offset"],
                                        "size": A["size"]}, "custom": c})

    # 4) Standalone low-quality block — degraded audio + altered transcript, so the
    #    cut has NO twin (unique at text AND audio level) and is labelled low quality:
    #    validates the unique-sample gate_drop path (report case 2).
    n_standalone = 0
    if args.standalone_lowq_count > 0:
        sperm = np.random.RandomState(args.seed + 6).permutation(n_base)
        for k in range(args.standalone_lowq_count):
            A = src[sperm[k % n_base]]
            out_id = _new_id(args.seed, "standalone", A["id"], k)
            prng = _rng(args.seed, "stdplan", out_id)
            ops, mev = plan_standalone(prng)
            mev["clean_ref_id"] = None          # no clean twin kept
            text, _ = edit_text(A["text"], 0.6, prng)   # heavy edit -> text-unique
            c = passthrough_custom(A)
            c["dedup_eval"] = {"is_synthetic": True, "dup_type": "standalone_lowq",
                               "original_id": A["id"], "expect_audio_dup_of": None,
                               "expect_text_dup_of": None,
                               "params": {"mos_family": mev["family"]}}
            c["mos_eval"] = mev
            items.append({"out_id": out_id, "text": text, "ops": ops,
                          "audio": {"tar_path": A["tar_path"], "offset": A["offset"],
                                    "size": A["size"]}, "custom": c})
            n_standalone += 1
        print(f"  standalone low-quality block: {n_standalone} cuts")

    rng.shuffle(items)
    print(f"  total output cuts: {len(items)}")

    # distribute round-robin across workers
    nworkers = max(1, args.num_workers)
    buckets = [[] for _ in range(nworkers)]
    for i, it in enumerate(items):
        buckets[i % nworkers].append(it)
    job_args = [(w, buckets[w], str(args.out_dir), assets, args.seed)
                for w in range(nworkers) if buckets[w]]

    print(f"  encoding with {len(job_args)} workers ...")
    gt_path = args.out_dir / "ground_truth.jsonl"
    with mp.get_context("spawn").Pool(len(job_args)) as pool, gt_path.open("w") as gt:
        for wid, rows in pool.imap_unordered(_worker, job_args):
            for r in rows:
                gt.write(json.dumps(r) + "\n")

    # merged shar_index.json
    cuts_field, rec_field = [], []
    for w in range(nworkers):
        rel = f"worker_{w:02d}"
        for p in sorted((args.out_dir / rel).glob("cuts.*.jsonl.gz")):
            cuts_field.append(f"{rel}/{p.name}")
        for p in sorted((args.out_dir / rel).glob("recording.*.tar")):
            rec_field.append(f"{rel}/{p.name}")
    (args.out_dir / "shar_index.json").write_text(json.dumps(
        {"version": 1, "fields": {"cuts": cuts_field, "recording": rec_field}}, indent=2))

    summary = {"src_dirs": [str(d) for d in args.src_dirs], "out_dir": str(args.out_dir),
               "seed": args.seed, "n_base": n_base, "n_synth": n_synth,
               "n_mos": sum(mos_counts.values()), "n_standalone": n_standalone,
               "n_total": len(items),
               "extra_frac": args.extra_frac, "synth_counts": counts,
               "mos_count": args.mos_count, "mos_counts": mos_counts,
               "standalone_lowq_count": n_standalone,
               "musan_dir": str(args.musan_dir) if args.musan_dir else None,
               "rir_dir": str(args.rir_dir) if args.rir_dir else None,
               "num_workers_out": len(job_args)}
    (args.out_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"ground truth -> {gt_path}")
    print("DONE")


if __name__ == "__main__":
    main()
