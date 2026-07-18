"""DNSMOS Pro (JIT) — DNS-MOS / NISQA / VCC variants."""

from __future__ import annotations

import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


# DNSMOS Pro JIT was trained with these STFT params.
STFT_N_FFT = 320
STFT_HOP = 160
STFT_WIN = 320

DEFAULT_VARIANT = "nisqa"

_DNSMOS_PRO_URL = (
    "https://github.com/fcumlin/DNSMOSPro/raw/refs/heads/main/runs/{variant}/model_best.pt"
)


def build_dnsmos(cfg: dict, device: torch.device) -> torch.jit.ScriptModule:
    """Load DNSMOS Pro JIT, downloading the checkpoint if needed."""
    explicit = cfg.get("dnsmos_model_path")
    if explicit:
        path = Path(explicit)
    else:
        variant = str(cfg.get("dnsmos_variant", DEFAULT_VARIANT)).lower()
        cache_dir = Path(cfg.get("dnsmos_cache_dir", "./dnsmos_cache"))
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / f"dnsmos_pro_{variant}.pt"
        if not path.exists():
            import requests
            url = _DNSMOS_PRO_URL.format(variant=variant.upper())
            logger.info("Downloading DNSMOS Pro (%s) from %s", variant, url)
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            path.write_bytes(resp.content)

    model = torch.jit.load(str(path), map_location=device).eval()

    # freeze + optimize_for_inference: folds parameters and runs JIT
    # inference passes — meaningful launch-overhead win on this small CNN.
    if cfg.get("dnsmos_jit_optimize", True):
        try:
            with torch.inference_mode():
                model = torch.jit.freeze(model)
                model = torch.jit.optimize_for_inference(model)
        except Exception as exc:
            logger.warning("DNSMOS Pro optimize_for_inference failed: %s", exc)
    return model
