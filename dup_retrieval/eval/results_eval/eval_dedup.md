# Dedup accuracy (make_eval_dataset schema)

======================================================================
TEXT dedup recall — cut shares its expect_text_dup_of's text cluster
======================================================================
  awgn               2160/2160 = 100.0%
  clean_control      1079/1079 = 100.0%
  clipping           1800/1800 = 100.0%
  crosstalk          1800/1800 = 100.0%
  degraded_quality   9940/9940 = 100.0%
  exact              3977/3977 = 100.0%
  music              2160/2160 = 100.0%
  naturalness        2160/2160 = 100.0%
  partial_crop       5965/5965 = 100.0%
  real_noise         2880/2880 = 100.0%
  reverb             2160/2160 = 100.0%
  same_text_diff_speaker 5965/5965 = 100.0%
  telephony          1800/1800 = 100.0%
  text_slight_change 65/5965 = 1.1%
  wrong_pairing      3977/3977 = 100.0%
  OVERALL            47888/53788 = 89.0%

======================================================================
AUDIO dedup recall — cut shares its expect_audio_dup_of's audio cluster
======================================================================
  awgn               1910/2160 = 88.4%
  clean_control      1073/1079 = 99.4%
  clipping           1625/1800 = 90.3%
  degraded_quality   9903/9940 = 99.6%
  exact              3966/3977 = 99.7%
  music              1559/2160 = 72.2%
  naturalness        1513/2160 = 70.0%
  partial_crop       1000/5965 = 16.8%
  real_noise         2553/2880 = 88.6%
  reverb             47/2160 = 2.2%
  telephony          1796/1800 = 99.8%
  text_slight_change 65/5965 = 1.1%
  wrong_pairing      0/3976 = 0.0%
  OVERALL            27010/46022 = 58.7%

======================================================================
AUDIO cluster purity — clusters whose members share one origin
======================================================================
  multi-member audio clusters: 26623
  pure clusters: 26623/26623 = 100.0%
  cuts in pure clusters: 54732/54732 = 100.0%
