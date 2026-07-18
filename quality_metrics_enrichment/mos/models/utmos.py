"""UTMOS v1 (``tarepan/SpeechMOS``) — neural MOS predictor."""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def build_utmos(cfg: dict, device: torch.device) -> torch.nn.Module:
    model = torch.hub.load(
        "tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True,
    )
    model = model.eval().to(device)
    if cfg.get("use_compile", False):
        try:
            model = torch.compile(model, mode="default", dynamic=True)
        except Exception as exc:
            logger.warning("torch.compile on UTMOS failed: %s", exc)
    return model
