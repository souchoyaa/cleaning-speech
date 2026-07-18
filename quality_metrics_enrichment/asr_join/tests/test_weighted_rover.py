"""Unit tests for the ref-anchored, weighted, deterministic ROVER consensus.

Proves the two properties the design must guarantee:
  1. DETERMINISM — the weights have distinct subset sums (unique arg-max for any
     vote split, incl. even voter counts), and the consensus is independent of
     hypothesis insertion order.
  2. MODERATE OVERRIDE POLICY — the original transcript (ref) is the anchor: any
     single ASR cannot override it, any 2-of-3 majority does, ref wins true 2-2
     splits, and ref insertions are preserved.

Pure stdlib (uses asr.rover's pure-Python alignment fallback) — runs locally
without rapidfuzz / pyarrow / torch:  python3 test_weighted_rover.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # quality_metrics_enrichment/

from asr_join import rover_offline  # noqa: E402

# The shipped eval weights.
WEIGHTS = {"ref": 1.5, "canary": 1.02, "parakeet": 1.01, "qwen": 1.00}


def mk(ref, parakeet, canary, qwen):
    """A merged row with single-word-comparable hypotheses."""
    return {
        "ref_text": ref,
        "hypotheses": {
            "parakeet": {"text": parakeet, "word_timestamps": []},
            "canary":   {"text": canary,   "word_timestamps": []},
            "qwen":     {"text": qwen,      "word_timestamps": []},
        },
    }


def enh(m):
    return rover_offline.consensus_enhanced(m, weights=WEIGHTS)


def text(m):
    return enh(m)["text"]


# ── 1. Determinism ───────────────────────────────────────────────────────────

def test_weights_are_tie_free():
    rover_offline.assert_deterministic_weights(WEIGHTS)  # must not raise


def test_tie_prone_weights_rejected():
    try:
        rover_offline.assert_deterministic_weights({"ref": 1.0, "a": 1.0, "b": 2.0})
    except ValueError:
        return
    raise AssertionError("expected ValueError for {1.0, 1.0, 2.0} (1+1 == 2)")


def test_independent_of_hypothesis_order():
    # Same votes, different dict insertion order -> identical consensus.
    a = {"ref_text": "a", "hypotheses": {
        "parakeet": {"text": "b"}, "canary": {"text": "b"}, "qwen": {"text": "c"}}}
    b = {"ref_text": "a", "hypotheses": {
        "qwen": {"text": "c"}, "canary": {"text": "b"}, "parakeet": {"text": "b"}}}
    assert text(a) == text(b) == "b"


def test_repeatable():
    m = mk("a", "b", "b", "c")
    assert text(m) == text(mk("a", "b", "b", "c"))


# ── 2. Moderate override policy ──────────────────────────────────────────────

def test_single_asr_cannot_override_ref():
    # ref + two ASRs agree on A; one ASR dissents (B) -> ref kept.
    assert text(mk("A", "B", "A", "A")) == "A"
    # All four disagree (1-1-1-1) -> ref anchors.
    assert text(mk("A", "B", "C", "D")) == "A"


def test_two_asr_majority_overrides_ref():
    # 2 ASRs agree on B (third dissents to C), no ASR backs ref -> B wins.
    assert text(mk("A", "B", "B", "C")) == "B"


def test_ref_wins_true_2v2_split():
    # ref=A, qwen backs A (2), parakeet+canary say B (2) -> ref's side wins.
    assert text(mk("A", "B", "B", "A")) == "A"


def test_unanimous_asr_overrides_ref():
    assert text(mk("A", "B", "B", "B")) == "B"


def test_ref_insertion_preserved():
    # ASRs deleted the leading word; ref keeps it.
    assert text(mk("A X", "X", "X", "X")) == "A X"


def test_enhanced_equals_ref_when_asrs_agree_with_ref():
    assert text(mk("hello world", "hello world", "hello world", "hello world")) \
        == "hello world"


# ── 3. Ambiguity record carries the deterministic default ────────────────────

def test_tie_records_default_for_itn():
    r = enh(mk("A", "B", "B", "A"))   # 2-2 split -> deterministic winner A
    amb = r["ambiguous_words"]
    assert len(amb) == 1, amb
    e = amb[0]
    assert e["default"] == "A"            # = the weighted winner (ref side)
    assert sorted(e["candidates"]) == ["A", "B"]


# ── 4. Case-insensitive voting (ASR casing is unreliable) ────────────────────

def test_case_only_agreement_keeps_original():
    # ref all-caps, every ASR lower-case -> same word, original casing kept.
    assert text(mk("EAT", "eat", "eat", "eat")) == "EAT"


def test_case_only_agreement_not_flagged_ambiguous():
    # A case-only difference is NOT a genuine disagreement.
    assert enh(mk("EAT", "eat", "eat", "eat"))["ambiguous_words"] == []


def test_genuine_override_still_wins_despite_case():
    # ref "EAT" but all ASRs genuinely say "ate" -> 3-of-3 majority overrides.
    assert text(mk("EAT", "ate", "ate", "ate")) == "ate"


def test_case_insensitive_in_sentence():
    # word 1: case-only agreement (keep ref casing); word 2: genuine override.
    assert text(mk("EAT Apples", "eat oranges", "eat oranges", "eat oranges")) \
        == "EAT oranges"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
