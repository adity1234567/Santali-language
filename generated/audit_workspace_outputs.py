from __future__ import annotations

import csv
import json
import math
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parent
FILE_SUMMARY_CSV = ROOT / "workspace_file_audit_summary.csv"
DATASET_SUMMARY_JSON = ROOT / "workspace_dataset_audit_summary.json"

SKIP_DIRS = {".venv", "__pycache__"}
GENERATED_REQUIRED = [
    "sentence_id",
    "santali_sentence",
    "english_gloss",
    "bangla_gloss",
    "sentence_type",
    "confidence",
    "source_eaf_file",
    "source_eaf_utterance_id",
    "source_eaf_pattern",
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


def should_skip(path: Path) -> bool:
    return any(part in SKIP_DIRS for part in path.parts)


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


def token_count(text: str) -> int:
    return len(re.findall(r"\S+", str(text or "")))


def has_mojibake(text: str) -> bool:
    return any(marker in str(text or "") for marker in ["Ã", "Â", "à¦", "à§", "É", "Ê", "Ì"])


def english_like_santali(text: str) -> bool:
    tokens = re.findall(r"\S+", str(text or ""))
    if len(tokens) < 3:
        return False
    ascii_words = re.findall(r"\b[a-zA-Z]{2,}\b", str(text or ""))
    return len(ascii_words) / max(len(tokens), 1) >= 0.5


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").lower()).strip()


def eaf_pairs() -> tuple[set[str], set[tuple[str, str]], dict[tuple[str, str], dict[str, str]]]:
    path = ROOT / "initial_data_eaf_utterances.csv"
    if not path.exists():
        return set(), set(), {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    files = {row["file"] for row in rows}
    pairs = {(row["file"], str(row["utterance_id"])) for row in rows}
    mapping = {(row["file"], str(row["utterance_id"])): row for row in rows}
    return files, pairs, mapping


def cited_pattern_matches(row: dict, eaf_map: dict[tuple[str, str], dict[str, str]]) -> bool:
    key = (str(row.get("source_eaf_file", "")), str(row.get("source_eaf_utterance_id", "")))
    eaf = eaf_map.get(key)
    if not eaf:
        return False
    pattern = normalize(row.get("source_eaf_pattern", ""))
    if not pattern:
        return False
    joined = " || ".join(normalize(eaf.get(column, "")) for column in EAF_MATCH_COLUMNS)
    return pattern in joined


def csv_summary(path: Path) -> dict:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    columns = reader.fieldnames or []
    summary = {
        "kind": "csv",
        "rows": len(rows),
        "columns": len(columns),
        "column_names": columns,
    }
    if "santali_sentence" in columns:
        sentences = [row.get("santali_sentence", "") for row in rows]
        summary.update(
            {
                "unique_santali_sentence": len(set(sentences)),
                "duplicate_santali_sentence": len(sentences) - len(set(sentences)),
                "blank_english_gloss": sum(is_blank(row.get("english_gloss")) for row in rows),
                "blank_bangla_gloss": sum(is_blank(row.get("bangla_gloss")) for row in rows),
                "blank_sentence_type": sum(is_blank(row.get("sentence_type")) for row in rows),
                "one_token_santali": sum(token_count(row.get("santali_sentence")) <= 1 for row in rows),
                "english_like_santali": sum(english_like_santali(row.get("santali_sentence", "")) for row in rows),
                "mojibake_rows": sum(has_mojibake(json.dumps(row, ensure_ascii=False)) for row in rows),
                "confidence_counts": dict(Counter(row.get("confidence", "") for row in rows)),
            }
        )
    if "generated_santali" in columns:
        sentences = [row.get("generated_santali", "") for row in rows]
        summary.update(
            {
                "unique_generated_santali": len(set(sentences)),
                "duplicate_generated_santali": len(sentences) - len(set(sentences)),
                "one_token_generated_santali": sum(token_count(row.get("generated_santali")) <= 1 for row in rows),
                "english_like_generated_santali": sum(english_like_santali(row.get("generated_santali", "")) for row in rows),
                "score_min": min((float(row.get("combined_score", 0) or 0) for row in rows), default=0),
                "score_max": max((float(row.get("combined_score", 0) or 0) for row in rows), default=0),
            }
        )
    return summary


def jsonl_summary(path: Path, eaf_files: set[str], eaf_pair_set: set[tuple[str, str]], eaf_map: dict[tuple[str, str], dict[str, str]]) -> dict:
    raw_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    rows = []
    permissive_errors = 0
    strict_errors = 0
    for line in raw_lines:
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            permissive_errors += 1
            continue
        try:
            strict_json_loads(line)
        except Exception:
            strict_errors += 1
    summary = {
        "kind": "jsonl",
        "physical_lines": len(raw_lines),
        "rows": len(rows),
        "permissive_json_errors": permissive_errors,
        "strict_json_errors": strict_errors,
    }
    if rows and any("santali_sentence" in row for row in rows):
        sentences = [str(row.get("santali_sentence", "")) for row in rows]
        ids = [str(row.get("sentence_id", "")) for row in rows]
        summary.update(
            {
                "unique_sentence_ids": len(set(ids)),
                "duplicate_sentence_ids": len(ids) - len(set(ids)),
                "unique_santali_sentence": len(set(sentences)),
                "duplicate_santali_sentence": len(sentences) - len(set(sentences)),
                "missing_required": {
                    key: sum(key not in row or is_blank(row.get(key)) for row in rows)
                    for key in GENERATED_REQUIRED
                },
                "confidence_counts": dict(Counter(str(row.get("confidence", "")) for row in rows)),
                "one_token_santali": sum(token_count(row.get("santali_sentence")) <= 1 for row in rows),
                "two_or_fewer_token_santali": sum(token_count(row.get("santali_sentence")) <= 2 for row in rows),
                "english_like_santali": sum(english_like_santali(row.get("santali_sentence", "")) for row in rows),
                "mojibake_rows": sum(has_mojibake(json.dumps(row, ensure_ascii=False)) for row in rows),
                "valid_source_eaf_file": sum(str(row.get("source_eaf_file", "")) in eaf_files for row in rows),
                "valid_source_file_utterance": sum((str(row.get("source_eaf_file", "")), str(row.get("source_eaf_utterance_id", ""))) in eaf_pair_set for row in rows),
                "source_pattern_verified": sum(cited_pattern_matches(row, eaf_map) for row in rows),
            }
        )
    return summary


def json_summary(path: Path) -> dict:
    try:
        obj = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception as exc:
        return {"kind": "json", "json_error": repr(exc)}
    summary = {"kind": "json", "top_type": type(obj).__name__}
    if isinstance(obj, dict):
        summary["keys"] = list(obj.keys())
        summary["preview"] = {key: obj[key] for key in list(obj.keys())[:10]}
    elif isinstance(obj, list):
        summary["items"] = len(obj)
        if obj and isinstance(obj[0], dict):
            summary["first_keys"] = list(obj[0].keys())
    return summary


def eaf_summary(path: Path) -> dict:
    root = ET.parse(path).getroot()
    tiers = root.findall(".//TIER")
    annotation_count = 0
    tier_counts = {}
    root_tiers = []
    for tier in tiers:
        tier_id = tier.attrib.get("TIER_ID", "")
        count = len(tier.findall(".//ANNOTATION"))
        annotation_count += count
        tier_counts[tier_id] = count
        if not tier.attrib.get("PARENT_REF"):
            root_tiers.append(tier_id)
    return {
        "kind": "eaf",
        "tiers": len(tiers),
        "annotations": annotation_count,
        "root_tiers": root_tiers,
        "tier_counts": tier_counts,
    }


def xlsx_summary(path: Path) -> dict:
    with ZipFile(path) as zf:
        names = zf.namelist()
        sheets = [name for name in names if name.startswith("xl/worksheets/sheet")]
    summary = {"kind": "xlsx", "worksheet_files": len(sheets)}
    if path.name == "MTZ_Santali_Dictionary.xlsx":
        try:
            import generate_santali_40k as gen

            entries = gen.read_dictionary(path)
            summary.update(
                {
                    "dictionary_entries": len(entries),
                    "unique_headwords": len({entry["hw"] for entry in entries if entry.get("hw")}),
                    "pos_counts": dict(Counter(entry.get("pos", "") for entry in entries).most_common(15)),
                }
            )
        except Exception as exc:
            summary["dictionary_error"] = repr(exc)
    return summary


def text_summary(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    return {
        "kind": path.suffix.lower().lstrip(".") or "text",
        "lines": text.count("\n") + 1,
        "chars": len(text),
        "mojibake_markers": has_mojibake(text),
    }


def py_summary(path: Path) -> dict:
    result = subprocess.run(
        [sys.executable, "-m", "py_compile", str(path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    summary = text_summary(path)
    summary["kind"] = "python"
    summary["compile_ok"] = result.returncode == 0
    summary["compile_error"] = result.stderr.strip()
    return summary


def summarize(path: Path, eaf_files: set[str], eaf_pair_set: set[tuple[str, str]], eaf_map: dict[tuple[str, str], dict[str, str]]) -> dict:
    rel = path.relative_to(ROOT).as_posix()
    base = {
        "file": rel,
        "size_bytes": path.stat().st_size,
        "suffix": path.suffix.lower(),
    }
    try:
        if path.suffix.lower() == ".csv":
            details = csv_summary(path)
        elif path.suffix.lower() == ".jsonl":
            details = jsonl_summary(path, eaf_files, eaf_pair_set, eaf_map)
        elif path.suffix.lower() == ".json":
            details = json_summary(path)
        elif path.suffix.lower() == ".eaf":
            details = eaf_summary(path)
        elif path.suffix.lower() == ".xlsx":
            details = xlsx_summary(path)
        elif path.suffix.lower() == ".py":
            details = py_summary(path)
        elif path.suffix.lower() in {".txt", ".md"}:
            details = text_summary(path)
        elif path.suffix.lower() in {".pdf", ".docx"}:
            details = {"kind": path.suffix.lower().lstrip("."), "read_mode": "metadata_only"}
        else:
            details = {"kind": "other"}
    except Exception as exc:
        details = {"kind": "error", "error": repr(exc)}
    base.update(details)
    return base


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    eaf_files, eaf_pair_set, eaf_map = eaf_pairs()
    files = [
        path
        for path in ROOT.rglob("*")
        if path.is_file() and not should_skip(path.relative_to(ROOT))
    ]
    summaries = [summarize(path, eaf_files, eaf_pair_set, eaf_map) for path in sorted(files)]

    DATASET_SUMMARY_JSON.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    with FILE_SUMMARY_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        fields = [
            "file",
            "kind",
            "size_bytes",
            "rows",
            "physical_lines",
            "columns",
            "strict_json_errors",
            "unique_santali_sentence",
            "duplicate_santali_sentence",
            "blank_english_gloss",
            "blank_bangla_gloss",
            "blank_sentence_type",
            "one_token_santali",
            "english_like_santali",
            "valid_source_file_utterance",
            "source_pattern_verified",
            "compile_ok",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)

    print(f"Audited files: {len(summaries)}")
    print(f"Wrote: {FILE_SUMMARY_CSV}")
    print(f"Wrote: {DATASET_SUMMARY_JSON}")
    for item in summaries:
        if item["file"] in {
            "santali_synthetic_40k_sentences.csv",
            "generated_santali_5k.jsonl",
            "generated_santali_5k_all_data.jsonl",
            "initial_data_eaf_utterances.csv",
            "MTZ_Santali_Dictionary.xlsx",
        }:
            print(item["file"], {k: item.get(k) for k in ["kind", "rows", "strict_json_errors", "unique_santali_sentence", "blank_english_gloss", "valid_source_file_utterance"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
