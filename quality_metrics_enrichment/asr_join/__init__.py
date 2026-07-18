"""Offline merge + ROVER — the 4th and final split-pipeline component.

Runs on the interactive node (no SLURM allocation needed). Reads per-rank
JSONLs from the 3 ASR runners' output trees, merges by cut_id, applies
ROVER + repetition + language-consistency filters, writes one consolidated
``rover/merged.jsonl`` per language.

Schema of one output row:
    {cut_id, duration, ref_text, speaker, language_hint, vad,
     hypotheses: {parakeet: {text, avg_logp, word_timestamps},
                  canary:   {text, avg_logp, word_timestamps, language},
                  qwen:     {text, avg_logp, language, error?}},
     rover: {primary, voting, text, primary_fallbacks,
             word_timestamps, timestamp_source,
             ambiguous_words: [{position, primary, candidates}, ...],
             text_itn?        # LLM-normalized form when rover.itn.enabled
            },
     language_consistency: {expected, detected, matches, all_consistent},
     filtered_reason: null | "all_failed" | "excess_repetition"}
"""
