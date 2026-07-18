# Pipeline evaluation — report building blocks

A **menu** of selectable evaluation components (matplotlib figures / markdown
tables / one-line sentences) for the dedup + quality pipeline. Point it at one
run, render the blocks you want, drop them into your report. Component-by-
component **and** end-to-end, with multilingual views.

## Quick start (in the container)

```bash
cd .../dup_retrieval/eval/report

# 1. See the menu (no run needed)
python3 build.py --list

# 2. Render every applicable block (GT-only blocks auto-skip without ground truth)
python3 build.py --run <RUN> --all --out report_out

# 3. Render a chosen subset, in YOUR order
python3 build.py --run <RUN> --out report_out \
    --blocks overview_summary,stage_funnel,dedup_recall_by_family,asr_model_wer
```

`<RUN>` can be any one of: the run dir (`.../dedup_out`), the config yaml
(reads `output_dir`), `run_manifest.json`, or `retention/assignments.parquet`.
`ground_truth.jsonl` and `rover/merged.jsonl` are auto-found next to the run
(override with `--gt` / `--rover`).

## Output (`--out` folder)

| file | what |
| --- | --- |
| `INDEX.md` | the full menu (every block: id, title, kind, flags, description) |
| `REPORT.md` | the selected blocks assembled in order, figures embedded |
| `<id>.png` | one figure per figure-block |
| `blocks/<id>.md` | each block's markdown on its own (copy-paste into your report) |

So: run `--all` once, browse `INDEX.md`, then either copy individual
`blocks/<id>.md` + `<id>.png` into your report, or re-run with `--blocks` in
your chosen order to get a ready `REPORT.md`.

## Blocks by component

* **overview** — run summary, corpus-reduction funnel (entirety).
* **manifest** — dataset/language composition, duration distribution.
* **text_dedup / audio_dedup** — cluster-size distributions, match-score
  distribution, one-line summaries.
* **dedup_accuracy** *(needs GT)* — TEXT vs AUDIO recall per family, audio
  cluster purity (false-merge proxy).
* **quality** — metric distributions, correlation heatmap, gate summary.
* **retention_accuracy** *(needs GT)* — Case 1 keep-best, Case 2 gate P/R/F1.
* **retention_outcome** — removed-by-reason, kept vs removed.
* **asr** — per-model WER + consensus + **enhanced** transcript, enhanced
  change-rate, before/after examples.
* **multilingual** — per-language overview (dedup % + low-quality %), per-language
  enhanced change-rate, language-consistency. *(degenerate but valid on a
  single-language run; meant for the multilingual scale-up.)*

## Extending

Add a function to `blocks.py` decorated with `@block(id, title, category, kind,
needs_gt=?, multilingual=?, desc=?)` returning `BlockResult(markdown=..., fig=...)`.
It appears in the menu and `--all` automatically. The loader (`loader.py`)
exposes everything a block needs as pandas DataFrames: `assignments`, `quality`,
`text_clusters`, `audio_clusters`, `rover`, `ground_truth`.
