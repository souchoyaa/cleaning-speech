# Results — what exists and where it lives

**All result data is on scratch, not in this repo**, under one common root:

```
$RUNS = /capstor/scratch/cscs/sgodey/data_selection_runs/
├── benchmark/     synthetic-eval runs (controlled benchmark)
├── dedup/         dedup runs on real corpora (per-corpus, combined, cross-dataset)
├── voxpopuli/     the 5-language end-to-end run
└── quality/       standalone MOS/ASR-pipeline outputs
```

Each run dir uses one convention:
```
$RUNS/<category>/<run>/
├── dedup_out/       dedup pipeline stages (manifest, text_dedup, fingerprint,
│                    audio_match, quality_ingest, retention, apply_to_shar) + run_manifest.json
├── quality_mos/     MOS scores — part_*.jsonl          (present when MOS ran)
└── quality_asr/     ASR outputs + rover/merged.jsonl   (present when the ASR ensemble ran)
```
(the stage layout inside `dedup_out/` is defined in `dup_retrieval/core/run_layout.py`.)

---

## Report element → result location (`$RUNS/…`)

| Report | Location | Config |
|---|---|---|
| **Tables 2–4** — benchmark (synthetic LibriSpeech) | `benchmark/eval/dedup_out` | `egs/eval.yaml` |
| **Table 4** — transcript-source (ROVER key) | `benchmark/eval/dedup_out_rovertext` | `egs/eval_rovertext.yaml` |
| LibriSpeech rover-key ablation | `benchmark/eval_clean/dedup/{ls_orig,ls_enh,ls_itn,ls_rover}` | `egs/dedup_lsclean_*.yaml` |
| **Table 3** — dedup-method comparison | `dedup/combined_mhperm` (default), `combined_bow`, `combined_bow_perm`, `combined` (baseline) | `egs/dedup_combined_*.yaml`, `dedup_english_combined.yaml` |
| **Table 6** — disjoint cross-dataset (MLS+PS+CV) | `dedup/combined_mhperm` | `egs/dedup_combined_mhperm.yaml` |
| **Table 7** — overlapping (MLS+LibriHeavy) | audio clusters in `dedup/libriheavy_mls_mn3`; fingerprints in `dedup/libriheavy_mls_mhperm` | `egs/dedup_libriheavy_mls_mhperm.yaml` |
| **Table 5** — VoxPopuli 5-lang end-to-end | `voxpopuli/{en,de,fr,es,it}` | `egs/voxpopuli_{lang}.yaml` |
| **MOS benchmark** — per-family AUROC (§B.2) | `quality/results` + in-repo `mos/eval/results_eval.txt`, `FINDINGS.md` | `mos/egs/synth.yaml` |
| **ASR/ROVER** enhancement (§B.1) | each run's `quality_asr/rover/merged.jsonl` | `asr_join/egs/*.yaml` |

Single-corpus runs (context for Tables 3/6): `dedup/{mls_en, mls_en_mhperm,
peoples_speech_en, peoples_speech_en_mhperm, cv_en_mhperm, libriheavy,
libriheavy_mhperm}`. VoxPopuli per-lang dedup ablations: `dedup/vp_{lang}_{ref,itn}`.

## Saved in-repo snapshots (small — kept in the repo)
- `dup_retrieval/eval/results_eval/` — `eval_dedup.md`, `eval_retention.md`,
  `RESULTS.md`, `retention_success.json`, `results_combo_experiment.txt`
- `quality_metrics_enrichment/mos/eval/` — `FINDINGS.md`, `results_eval.txt`,
  `results_combo_experiment.txt`, `results.txt`

---

## Disposable artifacts (reclaim scratch space)

Already removed: ~100 MB of `_smoke*`, `_probe`, `pipeline_out_*`, `mos_synth`, pilots.

Left in place (large — verify before deleting):

| Path | Size | Why disposable |
|---|---|---|
| `$S/dedup_pipeline_test/librispeech_eval` | 36 G | OLD benchmark, superseded by `benchmark/eval` |
| `$S/dedup_pipeline_test/voxpopuli_pilot_out` | 1.1 G | pilot, superseded by `voxpopuli/` |
| `$S/dedup_pipeline_test/synthetic_shar` | 644 M | regenerable via `make_eval_dataset` |
| `$RUNS/dedup/libriheavy_mls_mn`, `…_mn2` | ~6 G | superseded multi-node dev iterations (keep `…_mhperm` + `…_mn3`) |

`$S = /capstor/scratch/cscs/sgodey`. `$S/dedup_pipeline_test/assets` (24 G) is the
**input** staging (LibriSpeech/MUSAN/RIR for `make_eval_dataset`) — keep if you'll
rebuild the benchmark, else deletable.

```bash
S=/capstor/scratch/cscs/sgodey; RUNS=$S/data_selection_runs
rm -rf $S/dedup_pipeline_test/librispeech_eval $S/dedup_pipeline_test/voxpopuli_pilot_out \
       $S/dedup_pipeline_test/synthetic_shar $RUNS/dedup/libriheavy_mls_mn $RUNS/dedup/libriheavy_mls_mn2
```

## Backup & archive (off the submission tree)
- Pre-cleanup backup: `$S/data_selection_backup_20260628_pre_cleanup.tar.gz` (101 M)
- Archived non-submission material: `…/utils/data_selection_archive_20260628/`
