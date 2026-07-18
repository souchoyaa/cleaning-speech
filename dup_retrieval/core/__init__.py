"""Two-stage duplicate retrieval over Lhotse Shar audio datasets.

Pipeline (each stage produces parquet, can re-run independently):

    A: manifest          — cuts.jsonl.gz -> manifest_part_*.parquet
                           (+ tar-offset locator for random audio access)
    C: text_dedup       — MinHash + LSH + verified-pair single-linkage
                           (sole text "flag" first pass; subsumes exact dups
                            since identical normalized_text -> Jaccard 1.0)
    D: audio_fingerprint — Wang-2003 / Moshi constellation hash on mel spectrogram
                           (random-access reader via the Stage-A locator)
    E: audio_match       — intra-cluster Hough 1D voting (single-linkage)
    Q: quality_ingest    — flatten quality JSONLs -> quality/quality_flat.parquet
                           (the cuDF-joinable bridge from the QA branch)
    F: retention         — quality-aware selection: acoustic-dup collapse + text cap
                           + flag-only unique handling (low_quality / keep_reason)
    G: apply_to_shar     — thin clone injecting cut.custom = dedup + quality + keep

Composite primary key everywhere: (dataset, cut_id).

See plan: ~/.claude/plans/precious-booping-goblet.md
"""
