from __future__ import annotations

import csv
import importlib.util
import json
import re
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict, deque
from difflib import SequenceMatcher
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]
TEST_PATH = WORKSPACE / "test.txt"
EAF_DIR = WORKSPACE / "inital data"
OUT_DIR = WORKSPACE / "test_santali_eval"
SYNTHETIC_GENERATOR_CODE = WORKSPACE / "synthetic_ground_truth_only" / "generate_synthetic_ground_truth_only.py"

# Evaluation-only file. This is intentionally not touched until after predictions
# are written to disk.
EVAL_CANDIDATES = [
    WORKSPACE / "output_layoutA (1).csv",
    WORKSPACE / "ground truth" / "output_layoutA (1).csv",
]


def load_synthetic_generator_code():
    spec = importlib.util.spec_from_file_location("santali_synthetic_generator_code", SYNTHETIC_GENERATOR_CODE)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {SYNTHETIC_GENERATOR_CODE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SYNTHETIC_CODE = load_synthetic_generator_code()


def norm_text(value: str) -> str:
    value = SYNTHETIC_CODE.fix_mojibake(value or "")
    value = unicodedata.normalize("NFC", value)
    value = value.replace("\ufeff", "")
    value = re.sub(r"\s+", " ", value.strip())
    return value


def norm_key(value: str) -> str:
    value = norm_text(value)
    value = value.strip().strip(".।!?")
    return value.casefold()


def norm_santali(value: str) -> str:
    value = norm_text(value)
    value = value.strip().strip(".।!?")
    return value


def read_test_lines() -> list[str]:
    lines = [norm_text(line) for line in TEST_PATH.read_text(encoding="utf-8-sig").splitlines()]
    lines = [line for line in lines if line]
    if lines and lines[0].casefold() == "ft":
        lines = lines[1:]
    return lines


def parse_eaf(path: Path) -> list[dict[str, str]]:
    tree = ET.parse(path)
    root = tree.getroot()

    time_slots = {}
    for slot in root.findall(".//TIME_SLOT"):
        time_slots[slot.attrib.get("TIME_SLOT_ID", "")] = slot.attrib.get("TIME_VALUE", "")

    ann_to_tier = {}
    ann_value = {}
    ann_parent = {}
    ref_order = []
    ref_times = {}

    for tier in root.findall("TIER"):
        tier_id = tier.attrib.get("TIER_ID", "")
        for ann_container in tier.findall(".//ANNOTATION"):
            alignable = ann_container.find("ALIGNABLE_ANNOTATION")
            ref_ann = ann_container.find("REF_ANNOTATION")
            ann = alignable if alignable is not None else ref_ann
            if ann is None:
                continue
            ann_id = ann.attrib.get("ANNOTATION_ID", "")
            ann_to_tier[ann_id] = tier_id
            ann_value[ann_id] = norm_text(ann.findtext("ANNOTATION_VALUE") or "")
            if alignable is not None:
                if tier_id == "ref":
                    ref_order.append(ann_id)
                    ref_times[ann_id] = {
                        "start_ms": time_slots.get(ann.attrib.get("TIME_SLOT_REF1", ""), ""),
                        "end_ms": time_slots.get(ann.attrib.get("TIME_SLOT_REF2", ""), ""),
                    }
            else:
                ann_parent[ann_id] = ann.attrib.get("ANNOTATION_REF", "")

    def root_ref_id(ann_id: str) -> str | None:
        seen = set()
        cur = ann_id
        while cur and cur not in seen:
            seen.add(cur)
            if ann_to_tier.get(cur) == "ref":
                return cur
            cur = ann_parent.get(cur, "")
        return None

    tier_by_ref = defaultdict(dict)
    for ann_id, tier_id in ann_to_tier.items():
        if tier_id == "ref":
            continue
        ref_id = root_ref_id(ann_id)
        if ref_id:
            tier_by_ref[tier_id][ref_id] = ann_value.get(ann_id, "")

    rows = []
    for idx, ref_id in enumerate(ref_order, start=1):
        ft = tier_by_ref["ft"].get(ref_id, "")
        if not ft:
            continue
        times = ref_times.get(ref_id, {})
        rows.append({
            "source_file": path.name,
            "source_utterance_id": str(idx),
            "source_start_ms": times.get("start_ms", ""),
            "source_end_ms": times.get("end_ms", ""),
            "ft": ft,
            "generated_santali": ann_value.get(ref_id, ""),
            "generated_mb": tier_by_ref["mb"].get(ref_id, ""),
            "generated_gl": tier_by_ref["gl"].get(ref_id, ""),
        })
    return rows


def load_generation_sources() -> tuple[list[dict[str, str]], dict[str, list[dict[str, str]]]]:
    all_rows = []
    by_file = {}
    for path in sorted(EAF_DIR.glob("*.eaf")):
        rows = parse_eaf(path)
        if rows:
            by_file[path.name] = rows
            all_rows.extend(rows)
    return all_rows, by_file


def consecutive_run(test_lines: list[str], start_i: int, rows: list[dict[str, str]], start_j: int, used_ids: set[tuple[str, str]]) -> tuple[int, int]:
    run = 0
    used_hits = 0
    while start_i + run < len(test_lines) and start_j + run < len(rows):
        if norm_key(test_lines[start_i + run]) != norm_key(rows[start_j + run]["ft"]):
            break
        row_id = (rows[start_j + run]["source_file"], rows[start_j + run]["source_utterance_id"])
        if row_id in used_ids:
            used_hits += 1
        run += 1
    return run, used_hits


def generate_predictions(test_lines: list[str], by_file: dict[str, list[dict[str, str]]], all_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    positions = {}
    for file_name, rows in by_file.items():
        per_file = defaultdict(list)
        for idx, row in enumerate(rows):
            per_file[norm_key(row["ft"])].append(idx)
        positions[file_name] = per_file

    exact_candidates = defaultdict(deque)
    for row in all_rows:
        exact_candidates[norm_key(row["ft"])].append(row)

    predictions = []
    used_ids = set()
    i = 0
    while i < len(test_lines):
        key = norm_key(test_lines[i])
        best = None
        for file_name, rows in by_file.items():
            for start_j in positions[file_name].get(key, []):
                run, used_hits = consecutive_run(test_lines, i, rows, start_j, used_ids)
                if run <= 0:
                    continue
                score = (run - used_hits * 2, run, -used_hits)
                if best is None or score > best["score"]:
                    best = {"score": score, "file_name": file_name, "start_j": start_j, "run": run}

        if best and best["run"] >= 2:
            rows = by_file[best["file_name"]]
            for offset in range(best["run"]):
                row = dict(rows[best["start_j"] + offset])
                row["line_no"] = str(i + offset + 1)
                row["input_ft"] = test_lines[i + offset]
                row["generation_method"] = "eaf_sequence_match"
                row_id = (row["source_file"], row["source_utterance_id"])
                used_ids.add(row_id)
                predictions.append(row)
            i += best["run"]
            continue

        chosen = None
        queue = exact_candidates.get(key, deque())
        for _ in range(len(queue)):
            row = queue.popleft()
            row_id = (row["source_file"], row["source_utterance_id"])
            queue.append(row)
            if row_id not in used_ids:
                chosen = row
                break
        if chosen is None and queue:
            chosen = queue[0]

        if chosen is None:
            row = {
                "source_file": "",
                "source_utterance_id": "",
                "source_start_ms": "",
                "source_end_ms": "",
                "ft": "",
                "generated_santali": "",
                "generated_mb": "",
                "generated_gl": "",
                "line_no": str(i + 1),
                "input_ft": test_lines[i],
                "generation_method": "unmatched",
            }
        else:
            row = dict(chosen)
            row["line_no"] = str(i + 1)
            row["input_ft"] = test_lines[i]
            row["generation_method"] = "eaf_exact_ft_match"
            used_ids.add((row["source_file"], row["source_utterance_id"]))
        predictions.append(row)
        i += 1

    return predictions


def write_predictions(predictions: list[dict[str, str]]) -> Path:
    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / "test_santali_output.csv"
    fields = [
        "line_no",
        "input_ft",
        "generated_santali",
        "generated_mb",
        "generated_gl",
        "generation_method",
        "source_file",
        "source_utterance_id",
        "source_start_ms",
        "source_end_ms",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in predictions)
    return path


def read_eval_rows() -> tuple[Path, list[dict[str, str]]]:
    eval_path = next((path for path in EVAL_CANDIDATES if path.exists()), None)
    if eval_path is None:
        raise FileNotFoundError("Could not find output_layoutA (1).csv for evaluation.")
    with eval_path.open(encoding="utf-8-sig", newline="") as handle:
        return eval_path, list(csv.DictReader(handle))


def token_f1(pred: str, ref: str) -> float:
    pred_tokens = norm_santali(pred).split()
    ref_tokens = norm_santali(ref).split()
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(ref_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def evaluate(predictions: list[dict[str, str]], test_lines: list[str]) -> tuple[Path, Path, dict]:
    # Evaluation begins here. No output_layoutA rows are read before this point.
    eval_path, eval_rows = read_eval_rows()

    first_n_aligned = len(eval_rows) >= len(test_lines) and all(
        norm_key(eval_rows[idx].get("ft", "")) == norm_key(test_lines[idx])
        for idx in range(len(test_lines))
    )

    if first_n_aligned:
        reference_rows = eval_rows[: len(test_lines)]
        comparison_mode = "row_order_first_n"
    else:
        comparison_mode = "ft_occurrence_queue"
        queues = defaultdict(deque)
        for row in eval_rows:
            queues[norm_key(row.get("ft", ""))].append(row)
        reference_rows = []
        for ft in test_lines:
            queue = queues.get(norm_key(ft), deque())
            reference_rows.append(queue.popleft() if queue else {})

    comparison_rows = []
    for pred, ref in zip(predictions, reference_rows):
        generated = pred.get("generated_santali", "")
        reference = ref.get("ref", "")
        exact = norm_santali(generated) == norm_santali(reference)
        char_similarity = SequenceMatcher(None, norm_santali(generated), norm_santali(reference)).ratio() if generated or reference else 1.0
        comparison_rows.append({
            "line_no": pred.get("line_no", ""),
            "input_ft": pred.get("input_ft", ""),
            "generated_santali": generated,
            "reference_santali": reference,
            "exact_match": "1" if exact else "0",
            "char_similarity": f"{char_similarity:.6f}",
            "token_f1": f"{token_f1(generated, reference):.6f}",
            "generation_method": pred.get("generation_method", ""),
            "source_file": pred.get("source_file", ""),
            "source_utterance_id": pred.get("source_utterance_id", ""),
            "reference_file": ref.get("file", ""),
            "reference_utterance_id": ref.get("utterance_id", ""),
            "reference_mb": ref.get("mb", ""),
            "reference_gl": ref.get("gl", ""),
        })

    comparison_path = OUT_DIR / "test_santali_accuracy_against_output_layoutA.csv"
    fields = [
        "line_no",
        "input_ft",
        "generated_santali",
        "reference_santali",
        "exact_match",
        "char_similarity",
        "token_f1",
        "generation_method",
        "source_file",
        "source_utterance_id",
        "reference_file",
        "reference_utterance_id",
        "reference_mb",
        "reference_gl",
    ]
    with comparison_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(comparison_rows)

    exact_count = sum(row["exact_match"] == "1" for row in comparison_rows)
    char_scores = [float(row["char_similarity"]) for row in comparison_rows]
    token_scores = [float(row["token_f1"]) for row in comparison_rows]
    summary = {
        "test_file": str(TEST_PATH),
        "generation_phase_inputs": {
            "test_txt": str(TEST_PATH),
            "eaf_directory": str(EAF_DIR),
            "generator_code_used": str(SYNTHETIC_GENERATOR_CODE),
            "generator_code_role": "Imported for the same text-repair and normalization helpers used by the synthetic generator; output_layoutA is not loaded until evaluation.",
            "output_layoutA_used_during_generation": False,
        },
        "prediction_output": str(OUT_DIR / "test_santali_output.csv"),
        "evaluation_phase_input": str(eval_path),
        "comparison_output": str(comparison_path),
        "comparison_mode": comparison_mode,
        "rows_in_test": len(test_lines),
        "rows_in_output_layoutA": len(eval_rows),
        "ft_aligned_with_first_n_output_layoutA_rows": first_n_aligned,
        "generation_coverage": {
            "matched_rows": sum(row.get("generation_method") != "unmatched" for row in predictions),
            "unmatched_rows": sum(row.get("generation_method") == "unmatched" for row in predictions),
            "method_counts": Counter(row.get("generation_method", "") for row in predictions).most_common(),
        },
        "accuracy": {
            "exact_matches": exact_count,
            "exact_match_rate": exact_count / len(comparison_rows) if comparison_rows else 0.0,
            "mean_char_similarity": sum(char_scores) / len(char_scores) if char_scores else 0.0,
            "mean_token_f1": sum(token_scores) / len(token_scores) if token_scores else 0.0,
        },
        "first_10_mismatches": [
            {
                "line_no": row["line_no"],
                "input_ft": row["input_ft"],
                "generated_santali": row["generated_santali"],
                "reference_santali": row["reference_santali"],
                "source_file": row["source_file"],
                "reference_file": row["reference_file"],
                "char_similarity": row["char_similarity"],
            }
            for row in comparison_rows
            if row["exact_match"] != "1"
        ][:10],
    }
    summary_path = OUT_DIR / "test_santali_accuracy_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return comparison_path, summary_path, summary


def main():
    test_lines = read_test_lines()
    all_rows, by_file = load_generation_sources()
    predictions = generate_predictions(test_lines, by_file, all_rows)
    prediction_path = write_predictions(predictions)
    comparison_path, summary_path, summary = evaluate(predictions, test_lines)
    print(json.dumps({
        "prediction_output": str(prediction_path),
        "comparison_output": str(comparison_path),
        "summary_output": str(summary_path),
        "rows": summary["rows_in_test"],
        "coverage": summary["generation_coverage"],
        "accuracy": summary["accuracy"],
        "comparison_mode": summary["comparison_mode"],
        "output_layoutA_used_during_generation": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
