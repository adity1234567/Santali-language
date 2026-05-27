from __future__ import annotations

import csv
import sys
from pathlib import Path

from semantic_test_generated_against_eaf import (
    GENERATED_CSV,
    EAF_UTTERANCE_CSV,
    TfIdfIndex,
    TokenOverlapIndex,
    eaf_bangla,
    eaf_english,
    eaf_santali,
    english_word_terms,
    make_summary,
    read_csv,
    score_generated_row,
    unicode_tokens,
    write_csv,
)


ROOT = Path(__file__).resolve().parent
MATCHES_CSV = ROOT / "generated_vs_eaf_sentence_semantic_matches.csv"
SUMMARY_CSV = ROOT / "generated_vs_eaf_sentence_semantic_summary.csv"
HIGH_CONFIDENCE_CSV = ROOT / "generated_high_confidence_sentence_eaf.csv"


def looks_like_sentence_ground_truth(row: dict[str, str]) -> bool:
    filename = row.get("file", "").lower()
    if filename.startswith("lexicon"):
        return False
    english_terms = english_word_terms(eaf_english(row))
    # Keep full utterances and short but real grammatical examples like
    # "He is happy"; exclude one-word lexical entries such as "Cry".
    return len(english_terms) >= 2


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    generated_rows = read_csv(GENERATED_CSV)
    all_eaf_rows = read_csv(EAF_UTTERANCE_CSV)
    eaf_rows = [row for row in all_eaf_rows if looks_like_sentence_ground_truth(row)]

    english_docs = [eaf_english(row) for row in eaf_rows]
    bangla_docs = [eaf_bangla(row) for row in eaf_rows]
    santali_docs = [eaf_santali(row) for row in eaf_rows]

    english_word_index = TfIdfIndex(english_docs, english_word_terms)
    english_overlap_index = TokenOverlapIndex(english_docs, english_word_terms)
    bangla_overlap_index = TokenOverlapIndex(bangla_docs, unicode_tokens)
    santali_overlap_index = TokenOverlapIndex(santali_docs, unicode_tokens)

    matches = [
        score_generated_row(
            generated,
            eaf_rows,
            english_word_index,
            english_overlap_index,
            bangla_overlap_index,
            santali_overlap_index,
        )
        for generated in generated_rows
    ]

    fields = [
        "generated_id",
        "template_id",
        "generated_santali",
        "generated_english",
        "generated_bangla",
        "best_eaf_file",
        "best_eaf_utterance_id",
        "best_eaf_santali",
        "best_eaf_english",
        "best_eaf_bangla",
        "combined_score",
        "english_word_score",
        "english_overlap_score",
        "bangla_overlap_score",
        "santali_overlap_score",
        "confidence",
    ]

    write_csv(MATCHES_CSV, matches, fields)
    summary = make_summary(matches, generated_rows, eaf_rows)
    summary.append(
        {
            "metric": "sentence_ground_truth_filter",
            "value": f"{len(eaf_rows)} of {len(all_eaf_rows)} EAF rows kept",
            "status": "INFO",
            "detail": "Excluded lexicon files and one-word lexical-style glosses.",
        }
    )
    write_csv(SUMMARY_CSV, summary, ["metric", "value", "status", "detail"])
    write_csv(
        HIGH_CONFIDENCE_CSV,
        [row for row in matches if row["confidence"] == "high"],
        fields,
    )

    print(f"Sentence-level EAF rows kept: {len(eaf_rows):,} of {len(all_eaf_rows):,}")
    print(f"Wrote sentence-level matches: {MATCHES_CSV}")
    print(f"Wrote sentence-level summary: {SUMMARY_CSV}")
    for row in summary[:10]:
        print(f"{row['status']}: {row['metric']} = {row['value']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
