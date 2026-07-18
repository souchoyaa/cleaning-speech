"""In-flight batch state (CPU-side handle on tensors still being d2h-copied)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PendingBatch:
    """Inflight state for one batch whose d2h copies are still in flight."""
    event: "torch.cuda.Event"
    cut_ids: list
    cuts: list
    audio_secs: float
    eff_secs: list = field(default_factory=list)
    utmos_t: Optional["torch.Tensor"] = None
    stoi_t: Optional["torch.Tensor"] = None
    pesq_t: Optional["torch.Tensor"] = None
    si_sdr_t: Optional["torch.Tensor"] = None
    dns_t: Optional["torch.Tensor"] = None
    aes_CE_t: Optional["torch.Tensor"] = None
    aes_CU_t: Optional["torch.Tensor"] = None
    aes_PC_t: Optional["torch.Tensor"] = None
    aes_PQ_t: Optional["torch.Tensor"] = None
