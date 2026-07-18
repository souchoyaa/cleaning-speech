"""Apply ROVER + filters to one merged cut. Wrapped around asr.rover
primitives, with two local extensions:

  - ``include_ref_text``: treat the dataset's ``ref_text`` as an extra
    voter (off by default — many datasets have noisy ref text).
  - ``ambiguous_words``: when ≥2 candidates tie for the top vote count
    at a word position, record ``{position, primary, candidates}`` for
    later LLM disambiguation in the ITN pass.
"""

from __future__ import annotations

from collections import Counter
from typing import Optional

from asr.rover import (
    DEFAULT_REPETITION_MAX_COUNT,
    DEFAULT_REPETITION_NGRAM,
    DEFAULT_ROVER_PRIMARY,
    DEFAULT_ROVER_VOTING,
    has_excess_repetition,
    language_consistency,
    word_align_indices,
)


def _rover_with_ambiguity(
    text_hyps: dict,
    primary_words: list,
    primary_ts: Optional[list],
    *,
    primary: str,
    voting: str,
    confidences: Optional[dict] = None,
) -> tuple:
    """Re-implementation of ``asr.rover.rover_2of3`` that ALSO returns
    ambiguous-word positions.

    Comparison is case-insensitive (ASR casing is unreliable): candidates
    that differ only in case share one vote, and the emitted surface form
    prefers the primary's word — so when the consensus agrees with the
    original transcript up to case, the original casing is preserved.

    Ambiguity: a position is flagged when ≥2 distinct candidates tie
    for the top vote count (majority mode) or top weight sum
    (confidence mode). The downstream LLM is asked to pick the
    contextually best one.

    Returns ``(consensus_text, word_timestamps, primary_fallbacks,
    ambiguous_words)`` where ambiguous_words is
    ``[{"position": int, "primary": str, "candidates": [str, ...]}, ...]``.
    """
    if not primary_words:
        return "", [], 0, []

    others = {k: (v or "").split() for k, v in text_hyps.items() if k != primary}
    aligned = {
        name: [
            other_words[idx] if idx is not None else None
            for idx in word_align_indices(primary_words, other_words)
        ]
        for name, other_words in others.items()
    }

    # "weighted" uses FIXED per-voter weights (deterministic — see
    # assert_deterministic_weights); "confidence" uses avg_logp.  Both share the
    # summed-weight arg-max emission path.
    use_conf = (voting in ("confidence", "weighted") and confidences is not None)
    p_conf = float(confidences.get(primary, 0.0)) if use_conf else 0.0

    out_words: list = []
    out_ts: list = []
    fallbacks = 0
    ambiguous: list = []

    for i, p_w in enumerate(primary_words):
        # Voting is CASE-INSENSITIVE: ASR models are unreliable about casing, so
        # a case-only difference ("EAT" vs "eat") is NOT a real disagreement and
        # must not override the original. Candidates are grouped by casefold key;
        # the emitted SURFACE prefers the primary's word (the original transcript
        # when primary="ref"), so the original casing is kept whenever the
        # consensus agrees with it case-insensitively.
        #
        # Per position collect (surface, weight, is_primary). weight is the
        # per-voter weight in weighted/confidence mode, else 1.0 (a unit vote).
        contribs = [(p_w, p_conf if use_conf else 1.0, True)]
        for name in others:
            w = aligned[name][i]
            if w is not None:
                wt = float(confidences.get(name, 0.0)) if use_conf else 1.0
                contribs.append((w, wt, False))

        # Group by casefold key: summed weight, raw vote count, chosen surface.
        key_weight: dict = {}
        key_count: Counter = Counter()
        key_surface: dict = {}
        for w, wt, is_primary in contribs:
            k = w.casefold()
            key_weight[k] = key_weight.get(k, 0.0) + wt
            key_count[k] += 1
            # Primary's surface always wins its key (keeps the original casing);
            # other keys keep the first surface seen (deterministic: primary is
            # added first, then slots in dict order).
            if is_primary:
                key_surface[k] = w
            else:
                key_surface.setdefault(k, w)

        if use_conf:
            # Summed-weight arg-max over keys. Distinct-subset-sum weights keep
            # this unique (case-merging only unions disjoint voter subsets, whose
            # sums stay distinct) -> deterministic for ANY voter count.
            win_key = max(key_weight.items(), key=lambda kv: kv[1])[0]
            emitted = key_surface[win_key]
        else:
            # Majority: most-voted key wins iff >=2 voters back it, else primary.
            win_key, n = key_count.most_common(1)[0]
            emitted = key_surface[win_key] if n >= 2 else p_w

        out_words.append(emitted)

        # Genuine disagreement: >=2 distinct keys tie on raw vote count. Case
        # variants share a key, so casing differences are never flagged. Record
        # the deterministic winner as "default" so the LLM ITN pass can refine it
        # in context but fall back to it when unsure (hybrid floor).
        top = key_count.most_common()
        top_count = top[0][1]
        tied = [k for k, c in top if c == top_count]
        if len(tied) >= 2:
            ambiguous.append({"position": i, "primary": p_w, "default": emitted,
                              "candidates": sorted(key_surface[k] for k in tied)})

        # Primary-fallback counter — same definition as rover_2of3:
        # primary stood alone (no other voter agreed, case-insensitively).
        if emitted == p_w and not use_conf:
            p_key = p_w.casefold()
            agreed = any(
                aligned[name][i] is not None and aligned[name][i].casefold() == p_key
                for name in others
            )
            if not agreed:
                fallbacks += 1

        if primary_ts is not None and i < len(primary_ts):
            ts_i = primary_ts[i] or {}
            out_ts.append({"w": emitted, "s": ts_i.get("s"), "e": ts_i.get("e")})

    return " ".join(out_words), out_ts, fallbacks, ambiguous


def apply(
    merged: dict,
    *,
    primary: str = DEFAULT_ROVER_PRIMARY,
    voting: str = DEFAULT_ROVER_VOTING,
    timestamp_source: Optional[str] = None,
    repetition_ngram: int = DEFAULT_REPETITION_NGRAM,
    repetition_max_count: int = DEFAULT_REPETITION_MAX_COUNT,
    expected_language: Optional[str] = None,
    include_ref_text: bool = False,
    weights: Optional[dict] = None,
) -> dict:
    """Add ``rover`` / ``language_consistency`` / ``filtered_reason`` in place.

    Filter outcomes:
      - all hypotheses empty → ``filtered_reason="all_failed"``, rover.text=""
      - rover.text triggers ``has_excess_repetition`` → ``"excess_repetition"``
      - otherwise → ``filtered_reason=None``

    Timestamps:
      - prefer primary's word_timestamps
      - if primary has none and ``timestamp_source`` is set + that slot has
        timestamps, use them and record ``timestamp_source`` in ``rover``
      - otherwise ``word_timestamps=[]`` and ``timestamp_source="none"``

    ``include_ref_text``: when True AND merged.ref_text is non-empty, ref
    is added as an extra voter under slot name "ref". If ``primary``
    equals ``"ref"``, the consensus anchors on ref (ASRs are the others).

    Ambiguous words are stored in ``rover.ambiguous_words`` and consumed
    by the LLM ITN pass in ``main.py`` (which adds an instruction to
    pick the contextually best candidate per position).

    ITN (LLM-based) is applied as a separate async pass in ``main.py``
    after this function returns — it needs vLLM HTTP concurrency and
    can't run inside the synchronous merge+rover loop. The post-pass
    writes ``rover.text_itn`` next to verbatim ``rover.text``. Caveat:
    ``word_timestamps`` stays aligned to ``text`` only — not to
    ``text_itn`` (one token can replace several spoken words).
    """
    hyps = merged.get("hypotheses") or {}

    # Per-slot text view; optionally fold in ref_text as a 4th voter.
    text_hyps = {name: (h.get("text") or "") for name, h in hyps.items()}
    if include_ref_text:
        ref = (merged.get("ref_text") or "").strip()
        if ref:
            text_hyps["ref"] = ref

    if all(not t.strip() for t in text_hyps.values()):
        merged["rover"] = {
            "primary":            primary,
            "voting":             voting,
            "text":               "",
            "primary_fallbacks":  0,
            "word_timestamps":    [],
            "timestamp_source":   "none",
            "ambiguous_words":    [],
        }
        merged["language_consistency"] = language_consistency(hyps, expected_language)
        merged["filtered_reason"] = "all_failed"
        return merged

    # Resolve primary words + timestamps. Special-case primary="ref" so a
    # user can anchor on the ground truth and use ASRs as voters.
    if primary == "ref":
        p_text = (merged.get("ref_text") or "").strip()
        p_ts = None
    else:
        p_hyp = hyps.get(primary) or {}
        p_text = (p_hyp.get("text") or "").strip()
        p_ts = p_hyp.get("word_timestamps") or None
    p_words = p_text.split() if p_text else []

    confidences = None
    if voting == "confidence":
        confidences = {
            name: float(h.get("avg_logp") or 0.0) for name, h in hyps.items()
        }
        # ref_text has no confidence; give it a neutral weight equal to
        # the mean of the ASR slots so it doesn't dominate or vanish.
        if include_ref_text and "ref" in text_hyps:
            confs = [v for v in confidences.values() if v]
            confidences["ref"] = sum(confs) / max(len(confs), 1) if confs else 0.0
    elif voting == "weighted":
        # Fixed per-voter weights (NOT avg_logp).  Any voter absent from the map
        # gets 1.0; "ref" carries its anchor weight.  These weights are what make
        # emission deterministic — validate them once with
        # ``assert_deterministic_weights`` before the run.
        wmap = dict(weights or {})
        confidences = {name: float(wmap.get(name, 1.0)) for name in text_hyps}

    consensus_text, consensus_ts, fallbacks, ambiguous_words = _rover_with_ambiguity(
        text_hyps, p_words, p_ts,
        primary=primary, voting=voting, confidences=confidences,
    )

    # Timestamp fallback: if primary had no ts and the configured
    # timestamp_source slot does, swap them in. (Common case: primary=qwen
    # which doesn't emit word timestamps, but canary's NFA does.)
    ts_source = primary if p_ts else "none"
    if not p_ts and timestamp_source and timestamp_source in hyps:
        ts_slot = hyps.get(timestamp_source) or {}
        ts_slot_ts = ts_slot.get("word_timestamps")
        if ts_slot_ts:
            consensus_ts = ts_slot_ts
            ts_source = timestamp_source

    merged["rover"] = {
        "primary":            primary,
        "voting":             voting,
        "text":               consensus_text,
        "primary_fallbacks":  fallbacks,
        "word_timestamps":    consensus_ts or [],
        "timestamp_source":   ts_source if (consensus_ts or []) else "none",
        "ambiguous_words":    ambiguous_words,
    }
    merged["language_consistency"] = language_consistency(hyps, expected_language)

    if has_excess_repetition(
        consensus_text, n=repetition_ngram, max_count=repetition_max_count,
    ):
        merged["filtered_reason"] = "excess_repetition"
    else:
        merged["filtered_reason"] = None

    return merged


def assert_deterministic_weights(weights: dict) -> None:
    """Verify weighted-voting ``weights`` give a UNIQUE arg-max for every
    possible vote split — i.e. all non-empty subset sums are distinct.

    This is what guarantees deterministic output no matter how many ASR models
    voted (the "even voter count" 2-2 tie problem): if no two voter subsets sum
    to the same weight, the winning candidate is never tied.  Called once at
    setup so a mis-tuned weight set fails loudly instead of silently resolving
    ties by arbitrary slot order.  Cost is O(2^n) — fine for a handful of voters.
    """
    items = sorted(weights.items())
    vals = [float(v) for _, v in items]
    n = len(vals)
    seen: dict = {}
    for mask in range(1, 1 << n):
        s = round(sum(vals[i] for i in range(n) if mask & (1 << i)), 9)
        if s in seen:
            a = [items[i][0] for i in range(n) if seen[s] & (1 << i)]
            b = [items[i][0] for i in range(n) if mask & (1 << i)]
            raise ValueError(
                f"weighted-voting weights are NOT tie-free: voter sets {a} and "
                f"{b} both sum to {s}, so a {a}-vs-{b} split has no unique winner. "
                f"Use weights with distinct subset sums (e.g. ref=1.5, "
                f"qwen=1.00, parakeet=1.01, canary=1.02).")
        seen[s] = mask


def consensus_enhanced(
    merged: dict,
    *,
    weights: dict,
    timestamp_source: Optional[str] = None,
    repetition_ngram: int = DEFAULT_REPETITION_NGRAM,
    repetition_max_count: int = DEFAULT_REPETITION_MAX_COUNT,
    expected_language: Optional[str] = None,
) -> dict:
    """Ref-anchored, weighted, deterministic "improved" consensus.

    Anchors on the dataset's original transcript (``primary="ref"``) and folds
    the ASRs in as weighted voters, so the original is the default and is only
    overridden when the ASR weight outweighs it (see the YAML ``rover.enhanced``
    weights).  Returns the rover dict; does NOT mutate ``merged``'s main
    ``rover`` (runs on a shallow copy).  When ``ref_text`` is empty the caller
    should fall back to the plain ASR consensus.
    """
    mc = {**merged}
    apply(
        mc,
        primary="ref",
        voting="weighted",
        weights=weights,
        include_ref_text=True,
        timestamp_source=timestamp_source,
        repetition_ngram=repetition_ngram,
        repetition_max_count=repetition_max_count,
        expected_language=expected_language,
    )
    return mc.get("rover") or {}
