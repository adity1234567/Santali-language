from __future__ import annotations

import json
import sys
from pathlib import Path

from accuracy_check_generated_against_ground_truth import (
    GENERATED_FILES,
    ROOT,
    TfIdfIndex,
    TokenOverlapIndex,
    best_match,
    english_terms,
    load_generated,
    load_ground_truth,
    summarize_file,
    unicode_tokens,
    write_csv,
)


SUMMARY_CSV = ROOT / "generated_accuracy_against_ground_truth_sentence_only_summary.csv"
SUMMARY_JSON = ROOT / "generated_accuracy_against_ground_truth_sentence_only_summary.json"
TOP_MATCHES_CSV = ROOT / "generated_accuracy_against_ground_truth_sentence_only_top_matches.csv"
LOW_SAMPLE_CSV = ROOT / "generated_accuracy_against_ground_truth_sentence_only_low_sample.csv"


def sentence_like(row: dict[str, str]) -> bool:
    # Keep full sentences and short grammatical examples, but drop one-word
    # lexical/dictionary rows such as "Four" or "palmyra palm".
    return len(english_terms(row.get("gt_english", ""))) >= 2


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    gt_rows = [row for row in load_ground_truth() if sentence_like(row)]

    english_index = TfIdfIndex([row["gt_english"] for row in gt_rows], english_terms)
    english_overlap_index = TokenOverlapIndex([row["gt_english"] for row in gt_rows], english_terms)
    bangla_overlap_index = TokenOverlapIndex([row["gt_bangla"] for row in gt_rows], unicode_tokens)
    santali_overlap_index = TokenOverlapIndex([row["gt_santali"] for row in gt_rows], unicode_tokens)

    summary_rows = []
    top_matches = []
    low_samples = []

    for path in GENERATED_FILES:
        if not path.exists():
            continue
        generated_rows, strict_errors = load_generated(path)
        matches = [
            {
                "dataset_file": path.name,
                **best_match(
                    row,
                    gt_rows,
                    english_index,
                    english_overlap_index,
                    bangla_overlap_index,
                    santali_overlap_index,
                ),
            }
            for row in generated_rows
        ]
        summary_rows.append(summarize_file(path, generated_rows, strict_errors, matches))
        strict = [match for match in matches if match["strict_reviewable"] == "true"]
        strict.sort(key=lambda match: float(match["combined_score"]), reverse=True)
        top_matches.extend(strict[:300])
        low_samples.extend([match for match in matches if float(match["combined_score"]) < 0.30][:100])

    summary_fields = [
        "file",
        "rows",
        "strict_json_errors",
        "unique_ids",
        "unique_santali",
        "blank_santali",
        "blank_english",
        "blank_bangla",
        "one_token_santali",
        "two_or_fewer_token_santali",
        "english_like_santali",
        "mojibake_rows",
        "exact_santali_matches",
        "exact_english_matches",
        "exact_bangla_matches",
        "weak_auto_matches_score_ge_0_55",
        "strict_reviewable_score_ge_0_75_content_ge_2",
        "mean_score",
        "median_score",
        "p90_score",
        "max_score",
    ]
    match_fields = [
        "dataset_file",
        "generated_id",
        "generated_santali",
        "generated_english",
        "generated_bangla",
        "best_gt_source",
        "best_gt_id",
        "best_gt_santali",
        "best_gt_english",
        "best_gt_bangla",
        "combined_score",
        "english_word_score",
        "english_overlap_score",
        "bangla_overlap_score",
        "santali_overlap_score",
        "content_word_overlap",
        "content_jaccard",
        "exact_santali_match",
        "exact_english_match",
        "exact_bangla_match",
        "strict_reviewable",
        "weak_auto_match",
    ]

    write_csv(SUMMARY_CSV, summary_rows, summary_fields)
    SUMMARY_JSON.write_text(json.dumps(summary_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(TOP_MATCHES_CSV, top_matches, match_fields)
    write_csv(LOW_SAMPLE_CSV, low_samples, match_fields)

    print(f"Sentence-like ground-truth rows: {len(gt_rows):,}")
    print(f"Wrote summary: {SUMMARY_CSV}")
    print(f"Wrote top strict matches: {TOP_MATCHES_CSV} ({len(top_matches):,} rows)")
    for row in summary_rows:
        print(
            row["file"],
            "strict=", row["strict_reviewable_score_ge_0_75_content_ge_2"],
            "weak=", row["weak_auto_matches_score_ge_0_55"],
            "mean=", row["mean_score"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
