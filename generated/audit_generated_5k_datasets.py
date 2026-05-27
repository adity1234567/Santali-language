from __future__ import annotations

import csv
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import generate_santali_40k as dictionary_reader


ROOT = Path(__file__).resolve().parent
DATASETS = [
    ROOT / "generated_santali_5k_all_data.jsonl",
    ROOT / "generated_santali_5k.jsonl",
]
EAF_CSV = ROOT / "initial_data_eaf_utterances.csv"
DICT_XLSX = ROOT / "MTZ_Santali_Dictionary.xlsx"
SUMMARY_OUT = ROOT / "generated_5k_dataset_audit_summary.json"
ISSUES_OUT = ROOT / "generated_5k_dataset_audit_rows.csv"

REQUIRED = [
    "sentence_id",
    "santali_sentence",
    "english_gloss",
    "bangla_gloss",
    "sentence_type",
    "confidence",
    "source_eaf_file",
    "source_eaf_utterance_id",
    "source_eaf_pattern",
    "dictionary_slots",
    "synthetic",
    "needs_native_review",
]
EAF_MATCH_COLUMNS = [
    "ref",
    "mb",
    "ipa",
    "ipa_original",
    "ipa_normalized",
    "segmentation",
    "bangla",
    "english",
    "ft",
]


def strict_json_loads(line: str) -> dict:
    return json.loads(
        line,
        parse_constant=lambda constant: (_ for _ in ()).throw(
            ValueError(f"invalid JSON constant {constant}")
        ),
    )


def is_blank(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    text = str(value).strip()
    return text == "" or text.lower() == "nan"


def normalize(text: str) -> str:
    text = str(text or "").lower()
    return re.sub(r"\s+", " ", text).strip()


def token_count(text: str) -> int:
    return len(re.findall(r"\S+", str(text or "")))


def has_bad_unicode_marker(text: str) -> bool:
    text = str(text or "")
    return any(marker in text for marker in ["Ã", "Â", "à¦", "à§", "É", "Ê", "Ì"])


def english_like_santali(text: str) -> bool:
    tokens = re.findall(r"\S+", str(text or ""))
    if len(tokens) < 3:
        return False
    ascii_words = re.findall(r"\b[a-zA-Z]{2,}\b", str(text or ""))
    return len(ascii_words) / max(len(tokens), 1) >= 0.5


def load_eaf() -> tuple[set[str], set[tuple[str, str]], dict[tuple[str, str], dict[str, str]]]:
    with EAF_CSV.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    files = {row["file"] for row in rows}
    pairs = {(row["file"], str(row["utterance_id"])) for row in rows}
    mapping = {(row["file"], str(row["utterance_id"])): row for row in rows}
    return files, pairs, mapping


def source_pattern_matches_eaf(row: dict, eaf_map: dict[tuple[str, str], dict[str, str]]) -> bool:
    key = (str(row.get("source_eaf_file", "")), str(row.get("source_eaf_utterance_id", "")))
    eaf = eaf_map.get(key)
    if not eaf:
        return False
    pattern = normalize(row.get("source_eaf_pattern", ""))
    if not pattern:
        return False
    joined = " || ".join(normalize(eaf.get(column, "")) for column in EAF_MATCH_COLUMNS)
    return pattern in joined


def generated_sentence_in_cited_eaf(row: dict, eaf_map: dict[tuple[str, str], dict[str, str]]) -> bool:
    key = (str(row.get("source_eaf_file", "")), str(row.get("source_eaf_utterance_id", "")))
    eaf = eaf_map.get(key)
    if not eaf:
        return False
    sentence = normalize(row.get("santali_sentence", ""))
    if not sentence:
        return False
    joined = " || ".join(normalize(eaf.get(column, "")) for column in EAF_MATCH_COLUMNS)
    return sentence in joined


def dictionary_slot_stats(rows: list[dict], dictionary_headwords: set[str]) -> tuple[int, int, int, int]:
    parse_errors = 0
    slot_count = 0
    blank_headwords = 0
    headword_hits = 0
    for row in rows:
        slots = row.get("dictionary_slots", [])
        if isinstance(slots, str):
            try:
                slots = json.loads(slots)
            except Exception:
                parse_errors += 1
                slots = []
        if not isinstance(slots, list):
            parse_errors += 1
            slots = []
        for slot in slots:
            slot_count += 1
            headword = str(slot.get("headword", "")).strip() if isinstance(slot, dict) else ""
            if not headword:
                blank_headwords += 1
            elif headword in dictionary_headwords:
                headword_hits += 1
    return slot_count, parse_errors, blank_headwords, headword_hits


def audit_dataset(path: Path, eaf_files: set[str], eaf_pairs: set[tuple[str, str]], eaf_map: dict[tuple[str, str], dict[str, str]], dictionary_headwords: set[str]) -> tuple[dict, list[dict[str, str]]]:
    raw_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    rows = []
    strict_errors = []
    permissive_errors = []
    for line_no, line in enumerate(raw_lines, 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except Exception as exc:
            permissive_errors.append({"line": line_no, "error": repr(exc)})
            continue
        try:
            strict_json_loads(line)
        except Exception as exc:
            strict_errors.append({"line": line_no, "error": repr(exc)})

    issue_rows: list[dict[str, str]] = []
    valid_file = 0
    valid_pair = 0
    pattern_match = 0
    generated_match = 0
    for row in rows:
        issues = []
        key = (str(row.get("source_eaf_file", "")), str(row.get("source_eaf_utterance_id", "")))
        if key[0] in eaf_files:
            valid_file += 1
        else:
            issues.append("source_file_not_eaf")
        if key in eaf_pairs:
            valid_pair += 1
        else:
            issues.append("source_pair_not_found")
        if source_pattern_matches_eaf(row, eaf_map):
            pattern_match += 1
        else:
            issues.append("source_pattern_not_verified")
        if generated_sentence_in_cited_eaf(row, eaf_map):
            generated_match += 1

        if is_blank(row.get("english_gloss")):
            issues.append("blank_english_gloss")
        if is_blank(row.get("bangla_gloss")):
            issues.append("blank_bangla_gloss")
        if is_blank(row.get("sentence_type")):
            issues.append("blank_sentence_type")
        if token_count(row.get("santali_sentence", "")) <= 1:
            issues.append("one_token_output")
        elif token_count(row.get("santali_sentence", "")) <= 2:
            issues.append("two_token_output")
        if has_bad_unicode_marker(json.dumps(row, ensure_ascii=False)):
            issues.append("possible_mojibake")
        if english_like_santali(row.get("santali_sentence", "")):
            issues.append("english_like_santali_field")

        if issues:
            issue_rows.append(
                {
                    "dataset": path.name,
                    "sentence_id": str(row.get("sentence_id", "")),
                    "issues": ";".join(issues),
                    "santali_sentence": str(row.get("santali_sentence", "")),
                    "english_gloss": str(row.get("english_gloss", "")),
                    "source_eaf_file": str(row.get("source_eaf_file", "")),
                    "source_eaf_utterance_id": str(row.get("source_eaf_utterance_id", "")),
                    "confidence": str(row.get("confidence", "")),
                }
            )

    slot_count, slot_parse_errors, slot_blank, slot_hits = dictionary_slot_stats(rows, dictionary_headwords)
    ids = [str(row.get("sentence_id", "")) for row in rows]
    sentences = [str(row.get("santali_sentence", "")) for row in rows]
    summary = {
        "dataset": path.name,
        "physical_lines": len(raw_lines),
        "rows_parsed": len(rows),
        "permissive_json_errors": len(permissive_errors),
        "strict_json_errors": len(strict_errors),
        "strict_json_error_examples": strict_errors[:5],
        "unique_sentence_ids": len(set(ids)),
        "duplicate_sentence_ids": len(ids) - len(set(ids)),
        "unique_santali_sentences": len(set(sentences)),
        "duplicate_santali_sentences": len(sentences) - len(set(sentences)),
        "confidence_counts": dict(Counter(str(row.get("confidence", "")) for row in rows)),
        "missing_required_counts": {
            key: sum(1 for row in rows if key not in row or is_blank(row.get(key)))
            for key in REQUIRED
        },
        "one_token_outputs": sum(token_count(row.get("santali_sentence", "")) <= 1 for row in rows),
        "two_or_fewer_token_outputs": sum(token_count(row.get("santali_sentence", "")) <= 2 for row in rows),
        "english_like_santali_rows": sum(english_like_santali(row.get("santali_sentence", "")) for row in rows),
        "possible_mojibake_rows": sum(has_bad_unicode_marker(json.dumps(row, ensure_ascii=False)) for row in rows),
        "valid_source_eaf_file_rows": valid_file,
        "valid_source_file_utterance_rows": valid_pair,
        "source_pattern_verified_rows": pattern_match,
        "generated_sentence_found_in_cited_eaf_rows": generated_match,
        "dictionary_slot_count": slot_count,
        "dictionary_slot_parse_errors": slot_parse_errors,
        "dictionary_slot_blank_headwords": slot_blank,
        "dictionary_slot_headword_hits": slot_hits,
        "top_source_files": Counter(str(row.get("source_eaf_file", "")) for row in rows).most_common(15),
    }
    return summary, issue_rows


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    eaf_files, eaf_pairs, eaf_map = load_eaf()
    dictionary_entries = dictionary_reader.read_dictionary(DICT_XLSX)
    dictionary_headwords = {entry["hw"] for entry in dictionary_entries if entry.get("hw")}

    summaries = []
    issue_rows = []
    for path in DATASETS:
        summary, issues = audit_dataset(path, eaf_files, eaf_pairs, eaf_map, dictionary_headwords)
        summaries.append(summary)
        issue_rows.extend(issues)

    SUMMARY_OUT.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    with ISSUES_OUT.open("w", encoding="utf-8-sig", newline="") as handle:
        fields = [
            "dataset",
            "sentence_id",
            "issues",
            "santali_sentence",
            "english_gloss",
            "source_eaf_file",
            "source_eaf_utterance_id",
            "confidence",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(issue_rows)

    print(f"Wrote summary: {SUMMARY_OUT}")
    print(f"Wrote row issues: {ISSUES_OUT} ({len(issue_rows):,} rows)")
    for summary in summaries:
        print()
        print(summary["dataset"])
        print(" rows:", summary["rows_parsed"])
        print(" strict_json_errors:", summary["strict_json_errors"])
        print(" valid EAF pairs:", summary["valid_source_file_utterance_rows"])
        print(" blank English:", summary["missing_required_counts"]["english_gloss"])
        print(" blank Bangla:", summary["missing_required_counts"]["bangla_gloss"])
        print(" blank sentence_type:", summary["missing_required_counts"]["sentence_type"])
        print(" one-token outputs:", summary["one_token_outputs"])
        print(" English-like Santali rows:", summary["english_like_santali_rows"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
