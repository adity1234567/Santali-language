from __future__ import annotations

import csv
import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EAF_DIR = ROOT / "inital data"
GENERATED_CSV = ROOT / "santali_synthetic_40k_sentences.csv"

UTTERANCE_CSV = ROOT / "initial_data_eaf_utterances.csv"
ANNOTATION_CSV = ROOT / "initial_data_eaf_annotations_long.csv"
TEST_REPORT_CSV = ROOT / "generated_vs_eaf_test_report.csv"

TIER_COLUMNS = [
    "ref",
    "mb",
    "gl",
    "ft",
    "ipa",
    "bangla",
    "english",
    "ipa_original",
    "ipa_normalized",
    "segmentation",
    "gloss",
    "notes",
]

TIER_MAP = {
    "ref": "ref",
    "mb": "mb",
    "gl": "gl",
    "ft": "ft",
    "ipa": "ipa",
    "bangla": "bangla",
    "english": "english",
    "ipa-original": "ipa_original",
    "ipa-normalized": "ipa_normalized",
    "segmentation": "segmentation",
    "gloss": "gloss",
    "notes": "notes",
}

SANTALI_SOURCE_COLUMNS = [
    "ref",
    "mb",
    "ipa",
    "ipa_original",
    "ipa_normalized",
    "segmentation",
]


def clean_text(value: str | None) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def standard_tier(tier_id: str) -> str:
    return TIER_MAP.get(tier_id.strip().lower(), tier_id.strip().lower().replace("-", "_"))


def parse_time(value: str | None) -> str:
    return "" if value is None else value


def parse_eaf(path: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    root = ET.parse(path).getroot()
    time_slots = {
        item.attrib["TIME_SLOT_ID"]: parse_time(item.attrib.get("TIME_VALUE"))
        for item in root.findall(".//TIME_SLOT")
    }

    annotations: dict[str, dict[str, str]] = {}
    children: dict[str, list[str]] = defaultdict(list)
    order = 0

    for tier in root.findall(".//TIER"):
        tier_id = tier.attrib.get("TIER_ID", "")
        parent_tier = tier.attrib.get("PARENT_REF", "")
        linguistic_type = tier.attrib.get("LINGUISTIC_TYPE_REF", "")

        for wrapper in tier.findall("./ANNOTATION"):
            annotation = next(iter(wrapper), None)
            if annotation is None:
                continue

            annotation_id = annotation.attrib.get("ANNOTATION_ID", "")
            annotation_ref = annotation.attrib.get("ANNOTATION_REF", "")
            start_slot = annotation.attrib.get("TIME_SLOT_REF1", "")
            end_slot = annotation.attrib.get("TIME_SLOT_REF2", "")
            value_node = annotation.find("ANNOTATION_VALUE")
            value = clean_text(value_node.text if value_node is not None else "")

            record = {
                "file": path.name,
                "annotation_id": annotation_id,
                "annotation_type": annotation.tag,
                "tier_id": tier_id,
                "tier_std": standard_tier(tier_id),
                "parent_tier": parent_tier,
                "linguistic_type": linguistic_type,
                "parent_annotation_id": annotation_ref,
                "start_ms": time_slots.get(start_slot, ""),
                "end_ms": time_slots.get(end_slot, ""),
                "value": value,
                "_order": str(order),
            }
            annotations[annotation_id] = record
            if annotation_ref:
                children[annotation_ref].append(annotation_id)
            order += 1

    long_rows = [
        {key: value for key, value in record.items() if key != "_order"}
        for record in sorted(annotations.values(), key=lambda item: int(item["_order"]))
    ]

    root_ids = [
        annotation_id
        for annotation_id, record in sorted(
            annotations.items(), key=lambda item: int(item[1]["_order"])
        )
        if not record["parent_annotation_id"]
    ]

    utterance_rows: list[dict[str, str]] = []
    for utterance_index, root_id in enumerate(root_ids, 1):
        stack = [root_id]
        descendants: list[dict[str, str]] = []
        while stack:
            current_id = stack.pop(0)
            current = annotations[current_id]
            descendants.append(current)
            stack.extend(children.get(current_id, []))

        tier_values: dict[str, list[str]] = defaultdict(list)
        raw_tiers: dict[str, list[str]] = defaultdict(list)
        for item in sorted(descendants, key=lambda ann: int(ann["_order"])):
            if item["value"]:
                tier_values[item["tier_std"]].append(item["value"])
                raw_tiers[item["tier_id"]].append(item["value"])

        root_record = annotations[root_id]
        row = {
            "file": path.name,
            "utterance_id": str(utterance_index),
            "root_annotation_id": root_id,
            "root_tier": root_record["tier_id"],
            "start_ms": root_record["start_ms"],
            "end_ms": root_record["end_ms"],
        }
        for column in TIER_COLUMNS:
            row[column] = " | ".join(tier_values.get(column, []))
        row["all_tiers_json"] = json.dumps(raw_tiers, ensure_ascii=False, sort_keys=True)
        utterance_rows.append(row)

    return utterance_rows, long_rows


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def normalize_sentence(text: str) -> str:
    text = re.sub(r"\[[^\]]+\]", " ", text.lower())
    text = re.sub(r"[.,!?;:\"'“”‘’()[\]{}।]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokens(text: str) -> list[str]:
    normalized = normalize_sentence(text)
    normalized = re.sub(r"[-|/]", " ", normalized)
    return [token for token in normalized.split() if token]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def test_generated_against_eaf(utterance_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    generated_rows = read_csv_rows(GENERATED_CSV)

    eaf_source_sentences: set[str] = set()
    eaf_source_tokens: set[str] = set()
    for row in utterance_rows:
        for column in SANTALI_SOURCE_COLUMNS:
            value = row.get(column, "")
            if not value:
                continue
            eaf_source_sentences.add(normalize_sentence(value))
            eaf_source_tokens.update(tokens(value))

    generated_sentences = [row["santali_sentence"] for row in generated_rows]
    generated_unique = set(generated_sentences)
    generated_tokens = [token for sent in generated_sentences for token in tokens(sent)]
    generated_token_set = set(generated_tokens)
    exact_matches = sum(
        1 for sentence in generated_sentences if normalize_sentence(sentence) in eaf_source_sentences
    )
    overlapping_tokens = generated_token_set & eaf_source_tokens
    token_coverage = len(overlapping_tokens) / max(len(generated_token_set), 1)

    template_counts = Counter(row["template_id"] for row in generated_rows)
    source_file_counts = Counter(row["file"] for row in utterance_rows)

    report = [
        metric("eaf_files", len(source_file_counts), "PASS", "Number of parsed .eaf files."),
        metric("eaf_utterance_rows", len(utterance_rows), "PASS", "Root utterance rows written from EAF."),
        metric("generated_rows", len(generated_rows), "PASS" if len(generated_rows) == 40_000 else "FAIL", "Generated CSV row count."),
        metric("generated_unique_santali", len(generated_unique), "PASS" if len(generated_unique) == len(generated_rows) else "FAIL", "Unique generated Santali strings."),
        metric("generated_templates", len(template_counts), "PASS", "Number of generation templates represented."),
        metric("exact_generated_matches_in_eaf", exact_matches, "INFO", "Exact normalized sentence matches. Low/zero is expected for synthetic expansion."),
        metric("generated_token_types", len(generated_token_set), "INFO", "Unique token types in generated Santali column."),
        metric("eaf_token_types", len(eaf_source_tokens), "INFO", "Unique token types in EAF Santali-like tiers."),
        metric("overlapping_token_types", len(overlapping_tokens), "INFO", "Token types shared by generated CSV and EAF source tiers."),
        metric("generated_token_type_coverage", f"{token_coverage:.4f}", "INFO", "Shared generated token types divided by generated token types."),
    ]

    report.append(
        metric(
            "largest_eaf_files_by_utterances",
            "; ".join(f"{name}={count}" for name, count in source_file_counts.most_common(8)),
            "INFO",
            "Largest source files after flattening.",
        )
    )
    report.append(
        metric(
            "largest_generated_templates",
            "; ".join(f"{name}={count}" for name, count in template_counts.most_common(8)),
            "INFO",
            "Most frequent generated templates.",
        )
    )
    return report


def metric(name: str, value: object, status: str, detail: str) -> dict[str, str]:
    return {
        "metric": name,
        "value": str(value),
        "status": status,
        "detail": detail,
    }


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if not EAF_DIR.exists():
        raise FileNotFoundError(EAF_DIR)
    if not GENERATED_CSV.exists():
        raise FileNotFoundError(GENERATED_CSV)

    eaf_files = sorted(EAF_DIR.glob("*.eaf"))
    if not eaf_files:
        raise RuntimeError(f"No .eaf files found in {EAF_DIR}")

    utterance_rows: list[dict[str, str]] = []
    annotation_rows: list[dict[str, str]] = []
    for path in eaf_files:
        file_utterances, file_annotations = parse_eaf(path)
        utterance_rows.extend(file_utterances)
        annotation_rows.extend(file_annotations)

    utterance_fields = [
        "file",
        "utterance_id",
        "root_annotation_id",
        "root_tier",
        "start_ms",
        "end_ms",
        *TIER_COLUMNS,
        "all_tiers_json",
    ]
    annotation_fields = [
        "file",
        "annotation_id",
        "annotation_type",
        "tier_id",
        "tier_std",
        "parent_tier",
        "linguistic_type",
        "parent_annotation_id",
        "start_ms",
        "end_ms",
        "value",
    ]
    report_fields = ["metric", "value", "status", "detail"]

    write_csv(UTTERANCE_CSV, utterance_rows, utterance_fields)
    write_csv(ANNOTATION_CSV, annotation_rows, annotation_fields)
    report_rows = test_generated_against_eaf(utterance_rows)
    write_csv(TEST_REPORT_CSV, report_rows, report_fields)

    print(f"Parsed .eaf files: {len(eaf_files)}")
    print(f"Wrote utterance CSV: {UTTERANCE_CSV} ({len(utterance_rows):,} rows)")
    print(f"Wrote annotation CSV: {ANNOTATION_CSV} ({len(annotation_rows):,} rows)")
    print(f"Wrote test report: {TEST_REPORT_CSV}")
    for row in report_rows[:10]:
        print(f"{row['status']}: {row['metric']} = {row['value']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
