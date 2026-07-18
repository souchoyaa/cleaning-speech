"""YAML-driven dispatcher for the 3 ASR runners.

Reads ``pipeline/egs/*.yaml`` and runs each enabled stage by shelling out to its
``launch.sh`` — no SLURM polling or joiner, so the per-component launch scripts
stay the single source of truth. ``stages.<name>.enabled: false`` (or CLI
``--stages parakeet,qwen``) selects the subset to run.

Multi-config qwen: ``qwen.configs: [...]`` runs the feeder once per config
sharing one vLLM server (the first config sets ``vllm_mode``; the rest reuse the
running server); ``qwen.config: cfg.yaml`` is the single-config shortcut.

Run::
    bash pipeline/scripts/launch.sh --config pipeline/egs/cv_fr.yaml [--stages qwen]
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


# Map stage name → relative path of its launch script under quality_metrics_enrichment/.
_LAUNCH_SCRIPTS = {
    "parakeet": "asr_parakeet/scripts/launch.sh",
    "canary":   "asr_canary/scripts/launch.sh",
    "qwen":     "asr_vllm/scripts/launch.sh",
}


def _resolve_config_path(config: str, qme_dir: Path) -> Path:
    """Allow either absolute paths or paths relative to QME_DIR in stage configs."""
    p = Path(config)
    return p if p.is_absolute() else qme_dir / p


def _collect_configs(stage_cfg: dict, stage_name: str) -> list:
    """Return list of configs from either ``config:`` or ``configs:``."""
    configs = stage_cfg.get("configs")
    if configs:
        if not isinstance(configs, list):
            raise ValueError(f"{stage_name}.configs must be a list")
        return list(configs)
    config = stage_cfg.get("config")
    if not config:
        raise ValueError(f"{stage_name} stage needs 'config' or 'configs'")
    return [config]


def _run_sbatch_stage(
    stage_name: str,
    stage_cfg: dict,
    qme_dir: Path,
) -> None:
    """parakeet / canary: invoke component launch.sh per config (sbatch + exit).

    Each call submits a SLURM job and returns immediately — we don't wait
    for completion. The user can ``squeue -u $USER`` afterward.
    """
    script = qme_dir / _LAUNCH_SCRIPTS[stage_name]
    for cfg in _collect_configs(stage_cfg, stage_name):
        cfg_path = _resolve_config_path(cfg, qme_dir)
        cfg_name = cfg_path.name
        logger.info("→ %s: sbatch --config %s", stage_name, cfg_name)
        subprocess.run(
            ["bash", str(script), "--config", cfg_name],
            check=True, env=os.environ.copy(),
        )


def _run_qwen_stage(stage_cfg: dict, qme_dir: Path) -> None:
    """qwen: chain through asr_vllm's launch.sh — synchronous feeder run.

    First config uses the declared ``vllm_mode`` (default mode_a: start
    new vLLM). Subsequent configs auto-switch to mode_b since the vLLM
    server brought up by the first feed is reused. This is how
    multi-language qwen on one server works.
    """
    script = qme_dir / _LAUNCH_SCRIPTS["qwen"]
    configs = _collect_configs(stage_cfg, "qwen")
    declared_mode = stage_cfg.get("vllm_mode", "mode_a")
    if declared_mode not in ("mode_a", "mode_b"):
        raise ValueError(
            f"qwen.vllm_mode must be 'mode_a' or 'mode_b' (got {declared_mode!r})",
        )

    # Walltime hygiene check — the #1 silent-failure mode at scale: vLLM
    # walltime expires mid-feed, feeder produces no transcripts for the
    # remaining hours. Warn if VLLM_TIME isn't set or looks short. We
    # can't easily compute "expected feeder duration" but we can flag
    # the missing-override case.
    vllm_time = os.environ.get("VLLM_TIME")
    if declared_mode == "mode_a":
        if not vllm_time:
            logger.warning(
                "⚠ VLLM_TIME not set — vllm_launcher will use its default walltime "
                "(typically 6h). If the feeder runs longer, vLLM dies mid-feed and "
                "the circuit breaker aborts. Set e.g. ``export VLLM_TIME=12:00:00`` "
                "before re-running, sized to your dataset.",
            )
        else:
            logger.info("vLLM walltime: %s (from $VLLM_TIME)", vllm_time)

    for i, cfg in enumerate(configs):
        cfg_path = _resolve_config_path(cfg, qme_dir)
        cfg_name = cfg_path.name
        # First iteration uses declared mode; later iterations always
        # reuse (mode_b) — the vLLM server is already up.
        effective_mode = declared_mode if i == 0 else "mode_b"
        args = ["bash", str(script), "--config", cfg_name]
        if effective_mode == "mode_b":
            args.append("--skip-vllm-launch")
        logger.info(
            "→ qwen [%d/%d]: %s  (mode=%s)",
            i + 1, len(configs), " ".join(args), effective_mode,
        )
        subprocess.run(args, check=True, env=os.environ.copy())


_STAGE_RUNNERS = {
    "parakeet": lambda sc, qme: _run_sbatch_stage("parakeet", sc, qme),
    "canary":   lambda sc, qme: _run_sbatch_stage("canary",   sc, qme),
    "qwen":     _run_qwen_stage,
}


# ---------------------------------------------------------------------------
# Batch mode — template + many datasets in one YAML.
# ---------------------------------------------------------------------------


def _render_template(node, vars):
    """Walk a YAML-loaded structure and substitute ``{var}`` in strings.

    Missing vars leave the placeholder intact (so partial renders don't
    blow up). Recurses into dicts and lists.
    """
    if isinstance(node, str):
        try:
            return node.format(**vars)
        except (KeyError, IndexError):
            return node
    if isinstance(node, dict):
        return {k: _render_template(v, vars) for k, v in node.items()}
    if isinstance(node, list):
        return [_render_template(v, vars) for v in node]
    return node


def _run_datasets_batch(
    cfg: dict, qme_dir: Path, requested_stages: list,
) -> None:
    """Iterate ``cfg['datasets']``: for each dataset, render each enabled
    stage's template with that dataset's vars, write the rendered yaml to
    ``render_dir/<name>/<stage>.yaml``, then dispatch via the standard
    per-stage runner.

    Common vars (output roots, etc.) are taken from the top-level cfg
    minus the reserved keys (``datasets``, ``stages``, ``render_dir``).
    Per-dataset ``vars`` block overrides common vars.

    The rendered files are kept on disk so reruns / debugging can inspect
    exactly what was submitted to each stage. Not deleted by us.
    """
    datasets = cfg.get("datasets") or []
    stages_cfg = cfg.get("stages") or {}
    if not datasets:
        logger.error("batch mode: 'datasets:' is empty")
        return
    if not stages_cfg:
        logger.error("batch mode: 'stages:' missing or empty")
        return

    render_dir = Path(cfg.get(
        "render_dir",
        str(qme_dir / "pipeline" / "egs" / "_rendered"),
    ))

    reserved = {"datasets", "stages", "render_dir"}
    common_vars = {k: v for k, v in cfg.items() if k not in reserved}

    # Stages to run for each dataset (CLI --stages > YAML enabled flags).
    if requested_stages:
        enabled_stages = [s for s in requested_stages if s in stages_cfg]
    else:
        enabled_stages = [
            name for name, sc in stages_cfg.items() if (sc or {}).get("enabled", True)
        ]

    unknown = [s for s in enabled_stages if s not in _STAGE_RUNNERS]
    if unknown:
        logger.error("batch: unknown stage(s) %s; valid: %s", unknown, sorted(_STAGE_RUNNERS))
        return

    logger.info(
        "batch mode: %d dataset(s), stages=%s, render_dir=%s",
        len(datasets), enabled_stages, render_dir,
    )

    # Track per-stage "first call" state so we can auto-downgrade qwen's
    # vllm_mode to mode_b after the first dataset (reuse the already-
    # running vLLM instead of spinning up N copies).
    seen_qwen_first = False

    for d in datasets:
        name = d.get("name")
        if not name:
            logger.error("dataset entry has no 'name' field, skipping: %r", d)
            continue
        d_vars = {"name": name, **common_vars, **(d.get("vars") or {})}

        for stage in enabled_stages:
            stage_cfg = stages_cfg.get(stage) or {}
            template_rel = stage_cfg.get("template")
            if not template_rel:
                logger.warning(
                    "dataset=%s stage=%s: no 'template' in stages.%s — skipping",
                    name, stage, stage,
                )
                continue
            template_path = qme_dir / template_rel
            if not template_path.is_file():
                logger.error(
                    "dataset=%s stage=%s: template not found: %s",
                    name, stage, template_path,
                )
                continue

            template = yaml.safe_load(template_path.read_text())
            rendered = _render_template(template, d_vars)

            # Write rendered YAML so the runner can read it and so we have
            # a permanent record of exactly what was submitted.
            out_path = render_dir / name / f"{stage}.yaml"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w") as f:
                yaml.safe_dump(rendered, f, sort_keys=False)

            # Build a synthetic stage_cfg the runner can consume: drop
            # 'template', inject the absolute config path.
            synth = {k: v for k, v in stage_cfg.items() if k != "template"}
            synth["config"] = str(out_path)

            # Across-dataset reuse for qwen: the first invocation brings
            # up vLLM (declared vllm_mode, default mode_a); every later
            # dataset reuses that server (mode_b) instead of spinning up
            # a new sml job per language.
            if stage == "qwen":
                if seen_qwen_first:
                    synth["vllm_mode"] = "mode_b"
                seen_qwen_first = True

            logger.info("=== dataset=%s stage=%s rendered=%s ===", name, stage, out_path)
            try:
                _STAGE_RUNNERS[stage](synth, qme_dir)
            except subprocess.CalledProcessError as e:
                logger.error(
                    "dataset=%s stage=%s FAILED (rc=%d) — continuing.",
                    name, stage, e.returncode,
                )
                continue
            logger.info("✓ dataset=%s stage=%s done", name, stage)

    logger.info("batch complete: %d dataset(s) processed", len(datasets))


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )


def _snapshot_sml_jobs(pattern: str = r"sml_") -> set:
    """Return the set of currently-running sml job IDs matching ``pattern``.

    Used by --shutdown-vllm-on-exit to tell apart sml jobs WE started
    (newcomers) from pre-existing ones (left alone)."""
    import re
    try:
        out = subprocess.run(
            ["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%i %j"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode != 0:
            return set()
        rx = re.compile(pattern)
        return {
            line.split()[0]
            for line in out.stdout.splitlines()
            if line.strip() and rx.search(line.split(maxsplit=1)[-1])
        }
    except Exception as e:
        logger.warning("could not snapshot sml jobs (%s) — shutdown will be a no-op", e)
        return set()


def main() -> int:
    ap = argparse.ArgumentParser(description="ASR pipeline orchestrator")
    ap.add_argument(
        "--config", required=True,
        help="YAML config (pipeline/egs/*.yaml shape).",
    )
    ap.add_argument(
        "--stages",
        help="Comma-separated stage names to run; overrides YAML enabled flags. "
             "Valid: parakeet, canary, qwen.",
    )
    ap.add_argument(
        "--reuse-vllm", action="store_true",
        help="Force qwen.vllm_mode=mode_b for ALL datasets (reuse existing "
             "vLLM server, never launch a new one). Overrides YAML setting.",
    )
    ap.add_argument(
        "--shutdown-vllm-on-exit", action="store_true",
        help="After all datasets are processed, scancel any sml jobs that "
             "weren't running when the orchestrator started. Safe to use "
             "with --reuse-vllm — pre-existing sml jobs are left alone.",
    )
    args = ap.parse_args()

    _setup_logging()

    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        print(f"Config not found: {cfg_path}", file=sys.stderr)
        return 1
    cfg = yaml.safe_load(cfg_path.read_text())

    # pipeline/main.py → quality_metrics_enrichment/
    qme_dir = Path(__file__).resolve().parent.parent
    if not (qme_dir / "asr_parakeet").is_dir():
        print(
            f"ERROR: derived QME_DIR={qme_dir} does not look like the "
            f"quality_metrics_enrichment tree (no asr_parakeet/ subdir).",
            file=sys.stderr,
        )
        return 1

    stages_cfg = cfg.get("stages") or {}
    if not stages_cfg:
        logger.error("pipeline config has no 'stages:' section")
        return 1

    # Apply CLI override: --reuse-vllm forces every qwen stage to mode_b
    # (skip sml launch, reuse an existing vLLM server).
    if args.reuse_vllm and "qwen" in stages_cfg:
        stages_cfg["qwen"] = {**(stages_cfg["qwen"] or {}), "vllm_mode": "mode_b"}
        logger.info("--reuse-vllm: forcing qwen.vllm_mode=mode_b for all datasets")

    # --shutdown-vllm-on-exit: snapshot pre-existing sml jobs before any
    # stage runs. On exit (success or crash) we scancel any sml job that
    # appeared during this run — i.e. things WE launched, never anyone
    # else's. Per-dataset stage launchers handle their own snapshots too,
    # but this catches sml jobs that survived across datasets (the common
    # case with --reuse-vllm: same server fronts many datasets).
    sml_before: set = set()
    if args.shutdown_vllm_on_exit:
        sml_before = _snapshot_sml_jobs()
        logger.info(
            "--shutdown-vllm-on-exit: snapshot of %d pre-existing sml job(s); "
            "newcomers will be scancel'd at exit.", len(sml_before),
        )

    def _shutdown_newcomers() -> None:
        if not args.shutdown_vllm_on_exit:
            return
        after = _snapshot_sml_jobs()
        newcomers = after - sml_before
        if not newcomers:
            logger.info("--shutdown-vllm-on-exit: no new sml jobs to cancel.")
            return
        for jid in sorted(newcomers):
            logger.info("--shutdown-vllm-on-exit: scancel %s", jid)
            try:
                subprocess.run(["scancel", jid], check=False, timeout=10)
            except Exception as e:
                logger.warning("scancel %s failed (%s)", jid, e)

    # Batch mode — many datasets, template-rendered. Triggered by
    # presence of top-level ``datasets:``. The CLI ``--stages`` still
    # narrows which stages run, just for all datasets.
    if cfg.get("datasets"):
        wanted = (
            [s.strip() for s in args.stages.split(",") if s.strip()]
            if args.stages else None
        )
        try:
            _run_datasets_batch(cfg, qme_dir, wanted or [])
        finally:
            _shutdown_newcomers()
        return 0

    if args.stages:
        wanted = [s.strip() for s in args.stages.split(",") if s.strip()]
    else:
        wanted = [name for name, sc in stages_cfg.items() if (sc or {}).get("enabled", True)]

    unknown = [s for s in wanted if s not in _STAGE_RUNNERS]
    if unknown:
        logger.error(
            "unknown stage(s): %s; valid: %s",
            unknown, sorted(_STAGE_RUNNERS),
        )
        return 1

    missing = [s for s in wanted if s not in stages_cfg]
    if missing:
        logger.error(
            "stage(s) requested but not declared in YAML: %s; declared: %s",
            missing, sorted(stages_cfg),
        )
        return 1

    logger.info("running %d stage(s): %s", len(wanted), wanted)
    try:
        for stage in wanted:
            stage_cfg = stages_cfg.get(stage) or {}
            try:
                _STAGE_RUNNERS[stage](stage_cfg, qme_dir)
            except subprocess.CalledProcessError as e:
                # One stage's failure shouldn't always poison the rest — sbatch
                # for parakeet may fail for transient cluster reasons while
                # canary is fine. Log and continue; user inspects squeue + logs.
                logger.error(
                    "stage %s FAILED (rc=%d): %s — continuing with remaining stages.",
                    stage, e.returncode, e,
                )
                continue
            logger.info("✓ stage %s done", stage)
    finally:
        _shutdown_newcomers()

    logger.info("orchestrator complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
