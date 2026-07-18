#!/usr/bin/env python3
"""Build report building-blocks from one pipeline run.

The blocks (see blocks.py) are a MENU: each renders a figure / table / sentence
about one pipeline component or the whole run.  You pick which ones and the
order; this driver renders them into an output folder you can lift into a report.

Usage
-----
  # See the menu (no run needed):
  python3 build.py --list

  # Render EVERY applicable block into ./report_out (GT blocks auto-skipped
  # if there's no ground_truth.jsonl):
  python3 build.py --run <run_dir|config.yaml|run_manifest.json> --all --out report_out

  # Render a chosen subset, in your order:
  python3 build.py --run <...> --out report_out \
      --blocks overview_summary,stage_funnel,dedup_recall_by_family,asr_model_wer

Outputs in --out:
  <id>.png          one figure per figure-block
  blocks/<id>.md    each block's markdown snippet (sentence / table) on its own
  INDEX.md          the full menu (id, title, kind, flags, description)
  REPORT.md         the selected blocks assembled in order (figures embedded)

A single --run path is enough: a run dir, the config yaml (reads output_dir),
run_manifest.json, or retention/assignments.parquet.  ground_truth.jsonl is
auto-found next to the run (or pass --gt).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from loader import PipelineReport            # noqa: E402
import blocks as B                           # noqa: E402

CATEGORY_ORDER = ["overview", "methodology", "manifest", "text_dedup", "audio_dedup",
                  "dedup_accuracy", "quality", "retention_accuracy",
                  "retention_outcome", "asr", "multilingual", "limitations"]


def _ordered_catalog():
    return sorted(B.CATALOG,
                  key=lambda b: (CATEGORY_ORDER.index(b.category)
                                 if b.category in CATEGORY_ORDER else 99, b.id))


def _flags(b):
    f = []
    if b.needs_gt:
        f.append("needs-GT")
    if b.multilingual:
        f.append("multilingual")
    return ("[" + ", ".join(f) + "]") if f else ""


def render_index() -> str:
    lines = ["# Report building blocks — menu\n",
             "Pick block ids and pass them (in your order) to "
             "`build.py --blocks a,b,c`.\n"]
    cur = None
    for b in _ordered_catalog():
        if b.category != cur:
            cur = b.category
            lines.append(f"\n## {cur}\n")
            lines.append("| id | title | kind | flags | description |")
            lines.append("| --- | --- | --- | --- | --- |")
        lines.append(f"| `{b.id}` | {b.title} | {b.kind} | {_flags(b)} | {b.desc} |")
    return "\n".join(lines) + "\n"


def cmd_list():
    print(render_index())


def build(source, block_ids, outdir: Path, gt=None, rover=None, rover_limit=None):
    rep = PipelineReport(source, ground_truth=gt, rover=rover, rover_limit=rover_limit)
    print(rep.describe(), file=sys.stderr)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "blocks").mkdir(exist_ok=True)
    (outdir / "INDEX.md").write_text(render_index())

    if block_ids is None:                    # --all: every block, skip GT-only if no GT
        block_ids = [b.id for b in _ordered_catalog()
                     if (rep.has_gt or not b.needs_gt)]

    parts = [f"# Pipeline evaluation report\n\n_{rep.describe().splitlines()[1]}_\n"]
    for bid in block_ids:
        b = B.BLOCKS.get(bid)
        if b is None:
            print(f"  ! unknown block: {bid}", file=sys.stderr)
            continue
        try:
            res = b.fn(rep)
        except Exception as e:               # one bad block never kills the report
            import traceback
            traceback.print_exc()
            res = B.BlockResult(markdown=f"_(error rendering `{bid}`: {e})_")
        section = [f"## {b.title}\n", res.markdown or ""]
        if res.fig is not None:
            png = outdir / f"{bid}.png"
            res.fig.savefig(png, bbox_inches="tight")
            B.plt.close(res.fig)
            section.append(f"\n![{b.title}]({bid}.png)")
        (outdir / "blocks" / f"{bid}.md").write_text("\n".join(section) + "\n")
        parts.append("\n".join(section))
        print(f"  ✓ {bid}", file=sys.stderr)

    (outdir / "REPORT.md").write_text("\n\n".join(parts) + "\n")
    print(f"\nWrote {outdir}/REPORT.md  (+ INDEX.md, blocks/, figures)", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", help="run dir | config.yaml | run_manifest.json | assignments.parquet")
    ap.add_argument("--out", type=Path, default=Path("report_out"))
    ap.add_argument("--blocks", help="comma-separated block ids, in your order")
    ap.add_argument("--all", action="store_true", help="render every applicable block")
    ap.add_argument("--list", action="store_true", help="print the block menu and exit")
    ap.add_argument("--gt", help="explicit ground_truth.jsonl (else auto-found)")
    ap.add_argument("--rover", help="explicit rover/merged.jsonl (else auto-found)")
    ap.add_argument("--rover-limit", type=int, help="cap rover rows parsed (speed)")
    args = ap.parse_args()

    if args.list:
        cmd_list()
        return
    if not args.run:
        ap.error("--run is required (or use --list)")
    ids = args.blocks.split(",") if args.blocks else None
    if ids is None and not args.all:
        ap.error("pass --blocks <ids> or --all")
    build(args.run, ids, args.out, gt=args.gt, rover=args.rover,
          rover_limit=args.rover_limit)


if __name__ == "__main__":
    main()
