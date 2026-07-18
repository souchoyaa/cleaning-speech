"""SQUIM Objective (torchaudio) — STOI / PESQ / SI-SDR predictor."""

from __future__ import annotations

import torch


def build_squim(cfg: dict, device: torch.device) -> torch.nn.Module:
    # SQUIM_OBJECTIVE is DPRNN-based; torch.compile gives no gain (graph breaks).
    from torchaudio.pipelines import SQUIM_OBJECTIVE
    return SQUIM_OBJECTIVE.get_model().eval().to(device)
