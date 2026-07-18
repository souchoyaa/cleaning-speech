# Unified output schema

One convention for every stage's output, across the dedup (`dup_retrieval`) and
quality (`quality_metrics_enrichment`) pipelines.  Single source of truth for the
dedup names: `dup_retrieval/core/run_layout.py`.

## Conventions
- **Sharded outputs:** `part_{NNNN}.{parquet|jsonl}`, one per rank/chunk.
- **IDs:** every row carries an explicit `dataset` + `cut_id`; the dedup quality
  join keys on `(dataset, cut_id)`.  No path-derived dataset tags.
- **Markers:** each dedup stage writes its completion payload (rows, counts,
  config hash) to both `_SUCCESS` and `stage.json` via
  `run_layout.finalize_stage`; the dedup run writes a top-level
  `run_manifest.json` indexing every stage (dir, completion, counts).
- **Format:** parquet for columnar/batch stages; jsonl for streaming inference
  (ASR/MOS, append + resume), both with the same `part_{NNNN}` naming.

## Dedup pipeline — `<output_dir>/`
```
run_manifest.json                       # index of stages
manifest/      part_{NNNN}.parquet   _SUCCESS
text_dedup/    clusters.parquet  candidate_edges.parquet   _SUCCESS
fingerprint/   part_{NNNN}.parquet   _SUCCESS
audio_match/   clusters.parquet  match_edges.parquet  huge_clusters.parquet  _SUCCESS
quality/       merged.parquet   _SUCCESS
retention/     assignments.parquet   _SUCCESS
output_shar/   <cloned shar>   _SUCCESS
```

## Quality pipeline — `<output_dir>/`
```
quality_asr/
  parakeet/    parakeet_rank_{NNNN}.jsonl
  canary/      canary_rank_{NNNN}.jsonl
  qwen/        qwen_rank_{NNNN}.jsonl
  rover/       merged.jsonl
quality_mos/   part_{NNNN}.jsonl; rows carry `dataset`
```

Both feed Stage-F retention via `quality_ingest`, which globs `merged.jsonl`
(ASR rover) + `part_*.jsonl` (MOS) under `retention.quality_search_paths` and
joins on the in-row `dataset` tag (`asr_join` and the MOS worker each stamp it).
No bridge/adapter copy — retention reads the native outputs directly.

## Naming rule
For every dedup stage, **stage name == module name == config key** (`manifest`,
`text_dedup`, `audio_fingerprint`, `audio_match`, `quality_ingest`, `retention`,
`apply_to_shar`).  Output *directories* are artifact names and may be shorter
(`audio_fingerprint` → `fingerprint/`, `quality_ingest` → `quality/`,
`apply_to_shar` → `output_shar/`); the mapping lives in
`dup_retrieval/core/run_layout.py`.