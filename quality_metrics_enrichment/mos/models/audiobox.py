"""AudioBox Aesthetics — CE / CU / PC / PQ scores on 10 s windows."""

from __future__ import annotations

import os
import sys


# Model's fixed window — do not change.
WIN_SEC = 10.0
DEFAULT_SUB_BATCH = 32

_AUDIOBOX_SRC = os.environ.get("AUDIOBOX_SRC")


def build_audiobox(cfg: dict):
    src = cfg.get("audiobox_src") or _AUDIOBOX_SRC
    if src and src not in sys.path:
        sys.path.insert(0, src)
    from audiobox_aesthetics.infer import AesPredictor
    return AesPredictor(checkpoint_pth=cfg.get("audiobox_model_path"), data_col="path")
