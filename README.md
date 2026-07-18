# Data-selection pipeline — deduplication + quality curation

> **Repository placement** — this repo is the `data_selection` component of the
> [audio-tokenisation pipeline](https://github.com/souchoyaa/benchmark-audio-tokenizer-w-dedup).
> To use it in-tree, place the repo root at
> `audio_tokenization/utils/data_selection/`. The `prepare_data/` folder is a copy of
> `audio_tokenization/utils/prepare_data/` (dataset -> Lhotse-shar conversion scripts;
> `prepare_parquet_to_shar.py` and `prepare_wds_to_shar.py` were extended for this
> project) and belongs at that sibling path.

Curates audio–transcript pairs for the Apertus audio-tokenisation stage. Two subsystems:

- **`dup_retrieval/`** — deduplication. Cheap text clustering (MinHash + LSH, char-12
  containment ≥ 0.8) proposes candidates on CPU; an audio-fingerprint check
  (constellation map + Hough voting) confirms them; quality-aware retention keeps the
  best member of each cluster. Also contains the synthetic-benchmark builder
  (`eval/`) and the LaTeX report builder (`eval/report/`).
- **`quality_metrics_enrichment/`** — quality signals. A 3-model ASR ensemble
  (Parakeet + Canary + Qwen3-ASR/Voxtral) merged by ROVER with an LLM ITN pass
  (`asr_*/`), and four neural MOS metrics (UTMOS, SQUIM, DNSMOS-Pro, AudioBox) (`mos/`).

Everything runs on **CSCS Clariden** inside the `quality-assesment-gh200` container
(one node = 4 GH200). Result data lives under `/capstor/scratch/cscs/sgodey/` — see
**[RESULTS.md](RESULTS.md)** for the full map.

---

## Running the dedup pipeline

Seven stages — `manifest → text_dedup → fingerprint → audio_match → quality_ingest →
retention → apply_to_shar` — driven by one orchestrator (`dup_retrieval/core/pipeline.py`).
`manifest`, `fingerprint`, and `audio_match` shard across the node's ranks/GPUs;
`text_dedup` (cuDF, single-GPU) and the retention stages run on rank 0.

Two checkouts are involved: `REPO` is the outer
[benchmark-audio-tokenizer-w-dedup](https://github.com/souchoyaa/benchmark-audio-tokenizer-w-dedup)
pipeline (the launch scripts set `PYTHONPATH` to it), and `DS` is **this repo**, placed
inside it at `audio_tokenization/utils/data_selection` as described above.

```bash
REPO=<path to your benchmark-audio-tokenizer-w-dedup checkout>
DS=$REPO/audio_tokenization/utils/data_selection   # this repo

# single node (4 ranks / 4 GPUs)
sbatch --export=ALL,CONFIG=$DS/dup_retrieval/egs/<cfg>.yaml,REPO_DIR=$REPO \
       $DS/dup_retrieval/core/submit.slurm

# multi-node (scales manifest / fingerprint / audio_match linearly)
sbatch --nodes=N --export=ALL,CONFIG=$DS/dup_retrieval/egs/<cfg>.yaml,REPO_DIR=$REPO \
       $DS/dup_retrieval/core/submit_mn.slurm
```

**Where outputs go.** Every config's `output_dir` points under one common root,
`/capstor/scratch/cscs/sgodey/data_selection_runs/{benchmark,dedup,voxpopuli,quality}/<run>/`.
A run dir holds `dedup_out/` (the pipeline stages — see `dup_retrieval/core/run_layout.py`:
`text_dedup/clusters.parquet`, `audio_match/clusters.parquet`,
`retention/assignments.parquet`, `run_manifest.json`) and, when the quality passes
ran, sibling `quality_mos/` and `quality_asr/` dirs. **[RESULTS.md](RESULTS.md)** maps
every report result to its exact run dir.

---

## Reproducing the report

Paths in the configs point at the SHAR datasets under
`/capstor/store/cscs/swissai/infra01/audio-datasets/SHAR/stage_2/`.

### Step 0 · create the synthetic benchmark (`eval`)
The controlled benchmark (Tables 2–4) is built by `eval/make_eval_dataset.py`: it takes
clean LibriSpeech and adds **labelled** synthetic families with known ground truth —
a *dedup block* (exact copy, slight-text-change, degraded-quality, partial-crop,
same-text-different-speaker, hard-negative) and a *quality block* (real/white noise,
music, reverb, naturalness/Griffin-Lim, crosstalk, clipping, telephony, clean control).
Every synthetic cut carries its family + source cut id in a custom field, so recall and
keep-best can be scored automatically.
```bash
python -m audio_tokenization.utils.data_selection.dup_retrieval.eval.make_eval_dataset \
    --src-dirs <librispeech-shar-dirs...> \
    --out-dir  /capstor/scratch/cscs/sgodey/data_selection_runs/benchmark/eval \
    --extra-frac 0.30 \        # dedup-block size, as a fraction of the clean base
    --mos-count  18000 \       # quality-block cuts
    --musan-dir  <MUSAN> --rir-dir <RIRS_NOISES> \   # real noise/music + RIRs (else synthetic fallback)
    --seed 1                   # reproducible
#   (build.py --help / make_eval_dataset.py --help list all knobs)
```

### Tables 2–4 · run + score the benchmark
```bash
# run dedup + MOS on the benchmark (egs/eval.yaml already points output_dir at it)
sbatch --export=ALL,CONFIG=$DS/dup_retrieval/egs/eval.yaml,REPO_DIR=$REPO \
       $DS/dup_retrieval/core/submit.slurm
# score vs ground truth + render the report blocks (tables/figures)
python -m ...dup_retrieval.eval.evaluate_dedup_eval  --run <run_dir>   # per-family recall
python -m ...dup_retrieval.eval.evaluate_retention   --run <run_dir>   # keep-best + gate
python  $DS/dup_retrieval/eval/report/build.py --run <run_dir> --all --out report_out
#   (build.py --list shows every block; --blocks a,b,c renders a subset)
```
**Table 4 (transcript-source effect):** rerun with `egs/eval_rovertext.yaml`, which
keys dedup on the ASR-*enhanced* transcript instead of the original.

### Table 3 · dedup-method comparison (MLS + People's Speech + Common Voice)
Run the four text-stage methods, then compare `audio_match` counts:
| method | config / script |
|---|---|
| baseline char-16 Jaccard | `eval/evaluate_dedup_eval.py` baseline / `egs/dedup_combined*.yaml` |
| BoW Jaccard ≥ 0.7 | `eval/bow_dedup.py` · `egs/dedup_combined_bow.yaml` |
| **MinHash-perm char-12 cont. ≥ 0.8 (default)** | `core/text_dedup.py` · `egs/dedup_combined_mhperm.yaml` |
| BoW-perm word-cont. ≥ 0.8 | `eval/bow_dedup_permissive.py` · `egs/dedup_combined_bow_perm.yaml` |

### Tables 6 & 7 · cross-dataset dedup
- **Disjoint (MLS + PS + CV):** `egs/dedup_combined_mhperm.yaml`
- **Overlapping (MLS + LibriHeavy, ~22 M cuts):** `egs/dedup_libriheavy_mls_mhperm.yaml` (multi-node)

### Table 5 · VoxPopuli end-to-end (5 languages, ~13 M cuts)
```bash
for lang in en de fr es it; do
  sbatch --export=ALL,CONFIG=$DS/dup_retrieval/egs/voxpopuli_${lang}.yaml,REPO_DIR=$REPO \
         $DS/dup_retrieval/core/submit.slurm
done
```

### MOS metric benchmark (§B.2 · per-family AUROC)
```bash
cd $DS/quality_metrics_enrichment
mos/scripts/launch.sh --config mos/egs/synth.yaml       # scores the labelled quality set
python mos/eval/analyze_mos_eval.py                     # → mos/eval/results_eval.txt (AUROC/family)
python mos/eval/analyze_retention.py                    # → results_combo_experiment.txt (keep-best combiner)
```

### ASR transcript enhancement (§B.1 · ROVER + Gemma ITN)
```bash
cd $DS/quality_metrics_enrichment
# needs a vLLM server for Qwen3-ASR (+ Gemma for ITN); see asr_vllm/scripts + tools_env/
pipeline/scripts/launch.sh   --config pipeline/egs/eval.yaml   # Parakeet + Canary + Qwen workers
asr_join/scripts/launch.sh   --config asr_join/egs/eval.yaml   # ROVER vote + LLM ITN → rover/merged.jsonl
```

---

## Tests (run inside the container)
```bash
python -m pytest dup_retrieval/tests -q                          # dedup algorithms + cuDF Stage-C backend
python -m pytest quality_metrics_enrichment/asr_join/tests -q    # weighted ROVER
```
