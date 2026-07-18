"""ASR pipeline orchestrator — single YAML, modular per-stage opt-out.

Dispatches to each component's existing ``launch.sh``:
  - parakeet → ``asr_parakeet/scripts/launch.sh`` (sbatch, fire-and-forget)
  - canary   → ``asr_canary/scripts/launch.sh``   (sbatch, fire-and-forget)
  - qwen     → ``asr_vllm/scripts/launch.sh``     (interactive feeder; brings
                                                   up vLLM in mode_a or mode_b)

The join step is NOT included — invoke ``asr_join/scripts/launch.sh``
manually once enough rows have landed.
"""
