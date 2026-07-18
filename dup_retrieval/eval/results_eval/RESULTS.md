# End-to-end dedup + MOS pipeline on the LibriSpeech eval benchmark

Full run of `dup_retrieval` (manifest → text_dedup → audio_fingerprint →
audio_match → quality_ingest → retention) with **MOS-only** quality, on the
labelled benchmark `librispeech_eval/en` (190,315 cuts: 132,553 LibriSpeech
base + 39,766 dedup synthetics + 18,000 MOS-quality cuts).

- Config: `dup_retrieval/egs/eval.yaml` (4×GH200, seed 1, 2026-06-23).
- Outputs: `…/librispeech_eval/dedup_out/`.
- Reproduce eval: `evaluate_dedup_eval.py` + `evaluate_retention.py`
  (`--final dedup_out/final/dedup.parquet --gt en/ground_truth.jsonl`).

## Pipeline run (timings)
manifest 190,315 cuts · Stage C (cudf) 91,851 cuts in 43,825 text clusters,
candidate_fraction 0.48, 6.6 s · Stage D random-access fingerprint (4 GPUs) ·
Stage E within-cluster matching **4 s** → 63,742 audio clusters · retention 3 s.
Retention: **162,206 kept / 28,109 dropped** (all acoustic-dup collapse);
19,031 flagged low_quality (10,817 kept-but-flagged).

> **Scale fix found & applied:** Stage E (`audio_match`) pickled the *entire*
> fingerprint set to every one of the 43,825 cluster tasks → stalled 30+ min on
> this dense set.  Fixed with a `ProcessPoolExecutor` initializer that ships the
> fingerprints to each worker **once**.  Stage E: 30+ min (stuck) → **4 s**.

## Dedup accuracy (`eval_dedup.md`)
- **TEXT recall ≈ 100%** for every family except `text_slight_change` (1.1%) —
  content edits are intentionally gated by the conservative 20×13 LSH banding
  (left to a lenient pass); overall 89.0% is this one family.
- **AUDIO cluster purity = 100%** (26,623 multi-member clusters, zero cross-origin
  merges) — the acoustic pass is a precision filter, exactly as designed.
- **AUDIO recall** tracks how much the audio actually changed: exact/codec/
  telephony 99%+, noise/clipping 88–90%, music/naturalness ~70%, reverb 2%,
  partial_crop 17%, wrong_pairing 0% (same audio + different text → text-first
  routing separates them: the documented limitation of the cheap-proxy design).

## Retention — the two report cases (`eval_retention.md`)

**CASE 1 — DUPLICATES (keep the best copy).** Of 6,891 audio clusters that
contain both a clean and a low-quality member, retention keeps the **clean**
member **98.4%** of the time; it keeps the low-quality member in **0.1%** (6
clusters).  Ranking = mean of z-scored active axes.

**CASE 2 — quality gate (unique samples).**  "Unique" = *acoustically alone*
(`cluster_size == 1`: no acoustic duplicate, whether or not the cut shares a text
cluster).  On the **4,824 genuinely-unique MOS cuts** the gate (`gate_score =
0.5·mean(z) + 0.5·min(z)`, bottom-`gate_percentile`%) gets precision 0.596,
**recall 0.994, F1 0.745** — per-family recall awgn/crosstalk/music 100%,
naturalness 99%, real_noise 99%, clipping 97% (telephony n=4 only).  Across ALL
17,999 MOS cuts (incl. acoustic dups) the flag is P 0.668 / R 0.794 / F1 0.725
(telephony 15% — the known hard band-limit case).  Only **3.9% false-positives
on truly-clean originals**.  `naturalness` ~99% is the key win — DNSMOS alone
misses it (clean-SNR but unnatural); the `min(z)` term lets UTMOS flag it.

> **`is_duplicate` semantics fixed.**  Originally a cut was "unique" only if it
> was in *no text cluster* (`audio_cluster_id < 0`) — so every degraded MOS cut,
> which keeps its reference's transcript, was text-clustered and thus never
> "unique", even reverb copies the fingerprint correctly did NOT acoustically
> match (2,112/2,160 reverb cuts are acoustically alone).  Redefined: a cut is a
> *duplicate* iff `cluster_size > 1`, *unique* iff `cluster_size == 1`.  Now Case
> 1 and Case 2 partition every cut, and the gate (and `gate_drop`) correctly
> reach acoustically-unique low-quality cuts.

## Metric combination (how it was chosen — `results_combo_experiment.txt`)
On the labelled benchmark, for the two cases:
- **keep-best:** `mean(z)` = 99.2% (beats DNSMOS-weighted-sum 98.5%, DNSMOS-alone
  90.1% which fails telephony 71%).  → ranking uses mean(z).
- **gate:** `0.5·mean + 0.5·min` = best balanced family coverage (AUROC 0.95).
  → gate uses mean+min.

## MOS-only vs ASR/both
This run is `quality_mode: mos`.  The benchmark has no ASR signal, so `both`
would be identical here (the ASR axis is simply absent).  `quality_mode ∈
{mos, asr, both}` is implemented and unit-tested (`test_selection.py`); ASR-only
and both can be measured once ASR scores are produced for an eval set.
