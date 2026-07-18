# MOS metric evaluation — which metric(s) to keep the best sample

Empirical study of how the four MOS metrics (UTMOS, SQUIM, DNSMOS Pro, AudioBox
Aesthetics) behave on a **labeled** test set, to decide how to combine them for
quality-aware selection.

- **Date:** 2026-06-23
- **Data:** synthetic dedup test set, `cv22_synth/en` (5257 cuts, 1000 families),
  built from `commonvoice22_sidon/en/validation`.
  `…/dedup_pipeline_test/synthetic_shar/cv22_synth/en`
- **MOS scores:** `…/dedup_pipeline_test/mos_synth/mos_rank_0000.jsonl`
  (all 4 metrics, `dnsmos_variant=nisqa`, run via `mos/egs/synth.yaml`).
- **Ground truth:** `…/synthetic_shar/cv22_synth/en/ground_truth.jsonl` —
  `clip` and low-SNR `noise` (5/12 dB) are labeled `expect_low_quality=true`;
  `original`/`exact`/`case`/`punct`/`text_edit` (clean audio), `speed` (warped),
  and `reverb` are **not** low quality.

Reproduce:

```bash
# 1. score the synthetic shar (single GPU, ~1 min)
srun --jobid=<JID> --overlap --ntasks=1 --gpus-per-task=1 \
  --environment=quality-assesment-gh200 --chdir=<QME> bash -c '
    export RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 PYTHONPATH=<QME>:<REPO>
    python -m mos.main --config <QME>/mos/egs/synth.yaml'
# 2. analyze (pure stdlib, runs on the login node)
python3 mos/eval/analyze_mos.py        # AUROC, correlation, combos
python3 mos/eval/analyze_retention.py  # family-level "keep best copy" test
```

---

## TL;DR

- **Best single metric: DNSMOS Pro (nisqa) and AudioBox-PQ** (AUROC 0.946 each).
- **AudioBox-PC is INVERTED** (AUROC 0.094): noise/clipping *raise* production
  complexity. PC is a *content descriptor*, not a quality axis — never sum it
  with a positive sign. (`retention.py` currently does, via `audiobox_OVL`.)
- **Naively averaging all sub-scores is worse than DNSMOS alone** (0.932 < 0.946):
  equal-weight blending dilutes the strong signal with weak/inverted ones.
- The 9 sub-scores collapse to **~3 independent axes** (perceptual quality /
  SNR-distortion / complexity); the rest are redundant (CU≈PQ at r=0.96).
- For the **within-cluster "pick the best copy"** decision it barely matters —
  every metric keeps a clean copy ~99% (the cluster contains the pristine
  original). DNSMOS alone is ~99% correct; a `min`/worst-axis rule is the only
  bad ranker (4% errors).

---

## 1. Discriminative power (AUROC, clean originals vs degraded)

AUROC = P(metric ranks a clean original above a degraded cut). 1.0 perfect,
0.5 useless, <0.5 inverted. Degraded = 724 `expect_low_quality` cuts.

| metric | AUROC | | metric | AUROC |
|---|---|---|---|---|
| **DNSMOS Pro (nisqa)** | **0.946** | | si_sdr | 0.845 |
| **AudioBox PQ** | **0.946** | | pesq | 0.836 |
| AudioBox CU | 0.923 | | stoi | 0.826 |
| AudioBox CE | 0.916 | | **AudioBox PC** | **0.094 ⚠** |
| UTMOS | 0.874 | | | |

Per degradation type (AUROC vs originals):

| aug | dnsmos | PQ | pesq | si_sdr | stoi | utmos | PC |
|---|---|---|---|---|---|---|---|
| clip (the hard case) | 0.87 | 0.87 | 0.59 | 0.62 | 0.58 | 0.69 | 0.23 |
| noise 5/12 dB | 1.00 | 1.00 | 1.00 | 0.99 | 0.99 | 0.99 | 0.00 |
| reverb (GT: *not* low-Q) | 1.00 | 0.89 | 1.00 | 1.00 | 1.00 | 1.00 | 0.33 |
| speed (GT: *not* low-Q) | 0.51 | 0.50 | 0.50 | 0.49 | 0.49 | 0.50 | 0.49 |

noise AUROC by SNR — DNSMOS/PQ/PESQ are robust, SI-SDR/STOI collapse at mild noise:

| SNR | dnsmos | PQ | pesq | si_sdr | stoi | utmos |
|---|---|---|---|---|---|---|
| 5 dB  | 1.00 | 1.00 | 1.00 | 0.99 | 0.99 | 0.99 |
| 12 dB | 0.99 | 1.00 | 1.00 | 0.96 | 0.94 | 0.96 |
| 20 dB | 0.99 | 0.99 | 0.99 | **0.69** | **0.69** | 0.82 |

Takeaways: **clip is the hard degradation** (only DNSMOS/PQ exceed 0.85);
**SI-SDR & STOI are weak at mild noise**; **speed is invisible to all** (correct);
**reverb is scored as very low quality by every perceptual metric** even though
GT calls it fine — a hard MOS gate WILL discard reverberant/real-room speech
(that is the profile knob, not a bug).

## 2. Combining naively hurts

| combiner | AUROC |
|---|---|
| DNSMOS only | **0.946** |
| mean z(all 9 sub-scores) | 0.932 |
| mean z(dnsmos, si_sdr, pesq, utmos) | 0.906 |
| mean z(dnsmos, si_sdr) | 0.924 |
| min z(dnsmos, si_sdr) | 0.908 |

Equal-weight averaging is the wrong combiner. Use one strong metric, or weight
**by discriminative power** — not uniformly.

## 3. Redundancy (Pearson r, all cuts) → ~3 real axes

- **Perceptual quality**: {utmos, pesq, dnsmos, PQ, CE, CU} all pairwise r > 0.7
  (CU≈PQ at r=0.96 — near-duplicates).
- **Intrusive / SNR**: {stoi, si_sdr} r = 0.89 with each other, weaker overall.
- **PC**: orthogonal / anti-correlated to all (r ≈ −0.6 … −0.84) — the
  complexity axis, not quality.

## 4. The actual retention decision ("keep best copy per family")

Top-1 kept member per duplicate family (783 families, avg size 6.4):

| scheme | % kept low-Q | % kept clean |
|---|---|---|
| current retention `0.4·utmos + 0.3·dnsmos + 0.2·aesOVL(incl PC)` | 0.1% | 99.9% |
| same but aesOVL **without PC** | 0.0% | 100.0% |
| dnsmos only | 0.6% | 99.4% |
| PQ only | 0.0% | 97.6% |
| z(dnsmos)+z(si_sdr)+z(utmos)+z(pesq) | 0.9% | 99.1% |
| **min** z(dnsmos,si_sdr,utmos,pesq) | **2.6%** | 97.4% |

Within a cluster it barely matters (a pristine copy is always present). The PC
bug is **latent** here because UTMOS+DNSMOS dominate the sum. The only clearly
bad ranker is the worst-axis `min` rule — good for *gating*, bad for *ranking*
near-identical copies.

---

## Recommendation

**Two uses need different combiners.**

1. **Within a duplicate cluster (pick the best copy):** rank by
   **DNSMOS Pro (primary), tie-break SI-SDR**. One metric suffices; do NOT use a
   `min`/worst-axis rule.

2. **Corpus-level filtering to a quality profile:** don't collapse to one
   scalar — apply a few **per-axis gates**, one per independent axis:
   - quality gate → **DNSMOS Pro** (strongest, reference-free); UTMOS optional 2nd opinion
   - distortion floor → **SQUIM SI-SDR** (cheap insurance, heavy degradation only)
   - scene descriptor → **AudioBox PC** to *select* a profile (clean = low PC,
     music/rich = high PC), **never as a penalty**.
   A worst-axis "reject if any gate fails" rule over these *diverse* axes is the
   principled gate (this noise-only test can't showcase it).

### Concrete `retention.py` fixes — APPLIED 2026-06-23

1. ✅ `audiobox_OVL` now = `mean(CE,CU,PQ)` — **PC dropped** (it inverted the
   signal). `_flatten_metrics`.
2. ✅ Each axis is **z-scored over the corpus before weighting** for *ranking*
   (`_norm_score`, used for keep-best + text-cap), so weights = relative
   influence. The per-dataset **floor keeps a raw, un-normalized score**
   (`_raw_score`) so its absolute config (e.g. 4.0) still means what it did.
3. ✅ Default weights flipped to **DNSMOS 0.40 ≥ UTMOS 0.30** (audiobox 0.20,
   rover 0.10); updated in `retention.py` + `egs/{full,debug,synth_test}.yaml`.

Not done (optional): SQUIM/SI-SDR is still absent from the blend — add an SI-SDR
floor term if you want a distortion gate, or leave SQUIM out to save compute.
Tests: `dup_retrieval/tests/test_selection.py` (PC-drop + z-score), all pass.

## Scope / caveat

This set simulates only **additive noise + clipping** (with reverb/speed/
formatting as controls). It does **not** cover TTS naturalness artifacts
(UTMOS's strength), codec/bandwidth loss, or background-music contamination
(AudioBox/PC's domain). So on this test the metrics are redundant and DNSMOS
alone suffices — but you keep a small diverse set (DNSMOS + SI-SDR floor + PC
descriptor) because real corpora have failure modes this set doesn't simulate.
See the dataset roadmap for closing those gaps.
