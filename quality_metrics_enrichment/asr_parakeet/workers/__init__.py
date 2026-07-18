"""Re-export the existing fused-pipeline workers, no copy.

Both modules are pure (no coordinator/joiner coupling) so they're
reusable here as-is. Pinning the import here keeps the rest of
``asr_parakeet`` decoupled from the layout of the legacy ``asr/``
package — if those move later, only this file changes.
"""

from asr.workers.parakeet import ParakeetWorker
from asr.workers.vad import (
    DEFAULT_MIN_SILENCE_MS as VAD_DEFAULT_MIN_SILENCE_MS,
    DEFAULT_MIN_SPEECH_MS as VAD_DEFAULT_MIN_SPEECH_MS,
    DEFAULT_THRESHOLD as VAD_DEFAULT_THRESHOLD,
    VadWorker,
)

__all__ = [
    "ParakeetWorker",
    "VadWorker",
    "VAD_DEFAULT_THRESHOLD",
    "VAD_DEFAULT_MIN_SPEECH_MS",
    "VAD_DEFAULT_MIN_SILENCE_MS",
]
