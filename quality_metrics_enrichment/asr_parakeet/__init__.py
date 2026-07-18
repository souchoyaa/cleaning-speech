"""Standalone Parakeet driver — one of three split-pipeline ASR runners.

Runs Parakeet (+ optional Frame-VAD) over a Lhotse Shar dataset on a
single GPU node (one rank per GPU via SLURM), writing per-rank
``parakeet_rank_NNNN.jsonl`` files that the offline ``asr_join``
package then merges with Canary and the vLLM-feeder outputs.

No remote calls, no ROVER, no alignment. Pure local NeMo forward.
"""
