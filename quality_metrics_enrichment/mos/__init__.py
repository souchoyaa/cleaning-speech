"""MOS quality-assessment pipeline — UTMOS, SQUIM, DNSMOS Pro, AudioBox.

Each metric is opt-in via cfg flags (``use_utmos``, ``use_squim``,
``use_dnsmos``, ``use_audiobox``). Audio is truncated to ``truncate_secs``
before any GPU work. Per-batch H2D runs on a dedicated transfer stream so
the next batch's upload overlaps with the current batch's d2h DMA.
"""
