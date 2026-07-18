"""ROVER 2-of-3 word voting + repetition + language consistency helpers.

Word alignment uses rapidfuzz's C++ Levenshtein opcodes.
"""

from __future__ import annotations

from collections import Counter
from typing import Optional

from rapidfuzz.distance import Levenshtein as _rf_levenshtein


DEFAULT_REPETITION_NGRAM = 15
DEFAULT_REPETITION_MAX_COUNT = 5
DEFAULT_ROVER_PRIMARY = "canary"
DEFAULT_ROVER_VOTING = "majority"


def word_align_indices(ref: list, hyp: list) -> list:
    """Return alignment indices: ``out[i] = j`` where ``hyp[j]`` aligns to
    ``ref[i]``, or ``None`` for a deletion in hyp.
    """
    n, m = len(ref), len(hyp)
    if n == 0:
        return []
    if m == 0:
        return [None] * n

    aligned: list = [None] * n
    for tag, s_st, s_end, d_st, d_end in _rf_levenshtein.opcodes(ref, hyp):
        if tag in ("equal", "replace"):
            k_max = min(s_end - s_st, d_end - d_st)
            for k in range(k_max):
                aligned[s_st + k] = d_st + k
    return aligned


def _word_align(ref: list, hyp: list) -> list:
    return [hyp[idx] if idx is not None else None
            for idx in word_align_indices(ref, hyp)]


def rover_2of3(
    hyps: dict,
    primary_words: list,
    primary_ts: Optional[list] = None,
    primary: str = DEFAULT_ROVER_PRIMARY,
    voting: str = DEFAULT_ROVER_VOTING,
    confidences: Optional[dict] = None,
) -> tuple:
    """Anchor on ``primary_words``, vote 2-of-3 with the others, carry timestamps.

    Returns ``(consensus_text, consensus_word_timestamps, primary_fallbacks)``.
    """
    if not primary_words:
        return "", [], 0

    others = {k: (v or "").split() for k, v in hyps.items() if k != primary}
    aligned = {name: _word_align(primary_words, words) for name, words in others.items()}

    use_conf = (voting == "confidence" and confidences is not None)
    p_conf = float(confidences.get(primary, 0.0)) if use_conf else 0.0

    out_words: list = []
    out_ts: list = []
    fallbacks = 0
    for i, p_w in enumerate(primary_words):
        if use_conf:
            weights: dict = {p_w: p_conf}
            for name in others:
                w = aligned[name][i]
                if w is None:
                    continue
                weights[w] = weights.get(w, 0.0) + float(confidences.get(name, 0.0))
            emitted = max(weights.items(), key=lambda kv: kv[1])[0]
        else:
            votes = [p_w]
            for name in others:
                w = aligned[name][i]
                if w is not None:
                    votes.append(w)
            word, n = Counter(votes).most_common(1)[0]
            emitted = word if n >= 2 else p_w

        out_words.append(emitted)

        # Count "primary alone" emissions consistently across modes.
        if emitted == p_w:
            agreed = any(
                aligned[name][i] == p_w
                for name in others
                if aligned[name][i] is not None
            )
            if not agreed:
                fallbacks += 1

        if primary_ts is not None and i < len(primary_ts):
            ts_i = primary_ts[i] or {}
            out_ts.append({"w": emitted, "s": ts_i.get("s"), "e": ts_i.get("e")})

    return " ".join(out_words), out_ts, fallbacks


def has_excess_repetition(
    text: str,
    n: int = DEFAULT_REPETITION_NGRAM,
    max_count: int = DEFAULT_REPETITION_MAX_COUNT,
) -> bool:
    tokens = text.split()
    if len(tokens) < n:
        return False
    counts = Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))
    return any(c >= max_count for c in counts.values())


def language_consistency(
    hypotheses: dict,
    expected_lang: Optional[str],
) -> dict:
    """Per-model language verdict for one clip.

    Falls back to cross-model consensus when ``expected_lang`` is None.
    """
    def _norm(x):
        if x is None:
            return None
        s = str(x).strip().lower()
        return s or None

    detected = {name: _norm(h.get("language")) for name, h in hypotheses.items()}
    expected = _norm(expected_lang)

    if expected is None:
        non_null = [v for v in detected.values() if v]
        if non_null:
            expected = Counter(non_null).most_common(1)[0][0]

    matches: dict = {}
    for name, lang in detected.items():
        if lang is None or expected is None:
            matches[name] = None
        else:
            matches[name] = (lang == expected)

    non_null = [v for v in matches.values() if v is not None]
    return {
        "expected": expected,
        "detected": detected,
        "matches": matches,
        "all_consistent": bool(non_null) and all(non_null),
    }
