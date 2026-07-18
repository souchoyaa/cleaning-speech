# Retention evaluation (190315 cuts)

kept=162206  dropped=28109  in-audio-cluster=54732  low_quality=19031
keep_reason: {'kept': 151389, 'low_quality': 10817, 'quality_dropped': 28109}

==============================================================================
CASE 1 — DUPLICATES: keep-best within audio clusters
==============================================================================
audio clusters: 63742
clusters with quality contrast (clean + low-quality member): 6891
  kept member is CLEAN:        6779/6891 = 98.4%
  kept member is LOW-QUALITY:  6/6891 = 0.1%  (errors)

==============================================================================
CASE 2 — UNIQUE samples (acoustically alone): quality gate
==============================================================================
Gate on UNIQUE (acoustically-alone) MOS cuts (n=4824, low-quality=2594, clean=2230):
  precision=0.596  recall=0.994  F1=0.745  (TP=2578 FP=1751 FN=16 TN=479)
  per-family recall:
    awgn           244/244 = 100%
    clipping       165/170 = 97%
    crosstalk      625/625 = 100%
    music          584/585 = 100%
    naturalness    643/647 = 99%
    real_noise     315/319 = 99%
    telephony      2/4 = 50%

Gate flag over ALL MOS cuts (incl. acoustic duplicates) (n=17999, low-quality=9485, clean=8514):
  precision=0.668  recall=0.794  F1=0.725  (TP=7531 FP=3747 FN=1954 TN=4767)
  per-family recall:
    awgn           1060/1060 = 100%
    clipping       659/919 = 72%
    crosstalk      866/866 = 100%
    music          1272/1279 = 99%
    naturalness    2117/2160 = 98%
    real_noise     1290/1401 = 92%
    telephony      267/1800 = 15%

false-positive rate on clean originals: 3.9% (target <= gate_percentile)
