"""Standalone Canary driver — split-pipeline ASR runner with inline NFA.

Runs Canary-1B-v2 (+ optional Frame-VAD + optional NFA forced alignment)
over a Lhotse Shar dataset on a single GPU node. Per-cut output:
``canary_rank_NNNN.jsonl`` with the Canary hyp; if alignment is enabled
its NFA-aligned word_timestamps replace Canary's CTC-aux ones in place
(no re-decoding audio in the offline join pass).

No remote calls. No ROVER. Pure local NeMo.
"""
