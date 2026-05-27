from __future__ import annotations

import csv
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
GROUND_TRUTH_DIR = ROOT / "ground truth"

GROUND_TRUTH_FILES = [
    GROUND_TRUTH_DIR / "Santali_dataset_parsed.csv",
    GROUND_TRUTH_DIR / "output_layoutA (1).csv",
    GROUND_TRUTH_DIR / "output_layoutB (1).csv",
]

GENERATED_FILES = [
    ROOT / "santali_synthetic_40k_sentences.csv",
    ROOT / "generated_santali.jsonl",
    ROOT / "generated_santali_large.jsonl",
    ROOT / "generated_santali_5k.jsonl",
    ROOT / "generated_santali_5k_all_data.jsonl",
    ROOT / "generated_santali_large.csv",
    ROOT / "generated_santali_5k.csv",
    ROOT / "generated_santali_5k_all_data.csv",
    ROOT / "generated_high_confidence_against_eaf.csv",
    ROOT / "generated_high_confidence_sentence_eaf.csv",
    ROOT / "generated_low_confidence_sample_against_eaf.csv",
]

GT_COMBINED_CSV = ROOT / "ground_truth_combined_for_accuracy.csv"
SUMMARY_CSV = ROOT / "generated_accuracy_against_ground_truth_summary.csv"
SUMMARY_JSON = ROOT / "generated_accuracy_against_ground_truth_summary.json"
TOP_MATCHES_CSV = ROOT / "generated_accuracy_against_ground_truth_top_matches.csv"
LOW_SAMPLE_CSV = ROOT / "generated_accuracy_against_ground_truth_low_sample.csv"

ENGLISH_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "because",
    "but",
    "by",
    "do",
    "does",
    "did",
    "for",
    "from",
    "have",
    "has",
    "he",
    "her",
    "his",
    "i",
    "in",
    "is",
    "it",
    "my",
    "not",
    "no",
    "of",
    "on",
    "one",
    "or",
    "please",
    "she",
    "that",
    "the",
    "there",
    "this",
    "to",
    "was",
    "we",
    "will",
    "with",
    "you",
    "me",
    "they",
    "them",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


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


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").replace("\ufeff", "")).strip()


def normalize_exact(text: str) -> str:
    text = normalize_space(text).lower()
    text = re.sub(r"[.,!?;:\"'“”‘’()[\]{}।|/\\]+", " ", text)
    return normalize_space(text)


def unicode_tokens(text: str) -> list[str]:
    text = normalize_exact(text)
    return [token for token in re.split(r"[\s\-]+", text) if token]


def english_terms(text: str) -> list[str]:
    terms = []
    for term in re.findall(r"[a-z]+(?:'[a-z]+)?", normalize_exact(text)):
        if term in ENGLISH_STOPWORDS or len(term) <= 1:
            continue
        if len(term) > 5 and term.endswith("ing"):
            term = term[:-3]
        elif len(term) > 4 and term.endswith("ed"):
            term = term[:-2]
        elif len(term) > 4 and term.endswith("s"):
            term = term[:-1]
        terms.append(term)
    return terms


def token_count(text: str) -> int:
    return len(re.findall(r"\S+", str(text or "")))


def english_like_santali(text: str) -> bool:
    tokens = re.findall(r"\S+", str(text or ""))
    if len(tokens) < 3:
        return False
    ascii_words = re.findall(r"\b[a-zA-Z]{2,}\b", str(text or ""))
    return len(ascii_words) / max(len(tokens), 1) >= 0.5


def has_mojibake(text: str) -> bool:
    return any(marker in str(text or "") for marker in ["Ã", "Â", "à¦", "à§", "É", "Ê", "Ì"])


class TfIdfIndex:
    def __init__(self, docs: list[str], analyzer):
        self.docs = docs
        self.analyzer = analyzer
        self.inverted: dict[str, list[tuple[int, float]]] = defaultdict(list)
        self.idf: dict[str, float] = {}
        self._build()

    def _build(self) -> None:
        counts_by_doc: list[Counter[str]] = []
        document_frequency: Counter[str] = Counter()
        for doc in self.docs:
            counts = Counter(self.analyzer(doc))
            counts_by_doc.append(counts)
            document_frequency.update(counts.keys())
        total_docs = len(self.docs)
        self.idf = {
            term: math.log((total_docs + 1) / (df + 1)) + 1.0
            for term, df in document_frequency.items()
        }
        for doc_id, counts in enumerate(counts_by_doc):
            weighted = {term: tf * self.idf[term] for term, tf in counts.items()}
            norm = math.sqrt(sum(value * value for value in weighted.values()))
            if norm == 0:
                continue
            for term, weight in weighted.items():
                self.inverted[term].append((doc_id, weight / norm))

    def query(self, text: str, top_k: int = 25) -> dict[int, float]:
        counts = Counter(self.analyzer(text))
        weighted = {
            term: tf * self.idf[term]
            for term, tf in counts.items()
            if term in self.idf
        }
        norm = math.sqrt(sum(value * value for value in weighted.values()))
        if norm == 0:
            return {}
        scores: defaultdict[int, float] = defaultdict(float)
        for term, weight in weighted.items():
            query_weight = weight / norm
            for doc_id, doc_weight in self.inverted.get(term, []):
                scores[doc_id] += query_weight * doc_weight
        return dict(sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k])


class TokenOverlapIndex:
    def __init__(self, docs: list[str], analyzer):
        self.analyzer = analyzer
        self.doc_terms = [set(analyzer(doc)) for doc in docs]
        self.inverted: dict[str, list[int]] = defaultdict(list)
        for doc_id, terms in enumerate(self.doc_terms):
            for term in terms:
                self.inverted[term].append(doc_id)

    def query(self, text: str, top_k: int = 25) -> dict[int, float]:
        query_terms = set(self.analyzer(text))
        if not query_terms:
            return {}
        counts: Counter[int] = Counter()
        for term in query_terms:
            counts.update(self.inverted.get(term, []))
        scores = {}
        for doc_id, overlap in counts.items():
            denom = math.sqrt(len(query_terms) * max(len(self.doc_terms[doc_id]), 1))
            scores[doc_id] = overlap / denom if denom else 0.0
        return dict(sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k])


def load_ground_truth() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in GROUND_TRUTH_FILES:
        if not path.exists():
            continue
        for index, row in enumerate(read_csv(path), 1):
            if path.name.startswith("Santali_dataset_parsed"):
                santali = row.get("santali", "")
                english = row.get("english", "")
                bangla = row.get("bangla", "")
                gt_id = str(index)
                extra = ""
            elif path.name.startswith("output_layoutA"):
                santali = row.get("ref", "")
                english = row.get("ft", "")
                bangla = ""
                gt_id = row.get("utterance_id", str(index))
                extra = row.get("gl", "")
            else:
                santali = row.get("ipa_normalized") or row.get("ipa_original") or row.get("segments", "")
                english = row.get("english", "")
                bangla = row.get("bangla", "")
                gt_id = row.get("utterance_id", str(index))
                extra = row.get("glosses", "")
            rows.append(
                {
                    "gt_source": path.name,
                    "gt_id": gt_id,
                    "gt_santali": normalize_space(santali),
                    "gt_english": normalize_space(english),
                    "gt_bangla": normalize_space(bangla),
                    "gt_extra": normalize_space(extra),
                }
            )
    return rows


def load_generated(path: Path) -> tuple[list[dict[str, str]], int]:
    strict_errors = 0
    rows: list[dict[str, str]] = []
    if path.suffix.lower() == ".jsonl":
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            try:
                strict_json_loads(line)
            except Exception:
                strict_errors += 1
            rows.append(item)
    elif path.suffix.lower() == ".csv":
        rows = read_csv(path)
    return rows, strict_errors


def generated_fields(row: dict[str, str]) -> tuple[str, str, str, str]:
    generated_id = str(row.get("sentence_id") or row.get("generated_id") or "")
    santali = str(row.get("santali_sentence") or row.get("generated_santali") or "")
    english = str(row.get("english_gloss") or row.get("generated_english") or "")
    bangla = str(row.get("bangla_gloss") or row.get("generated_bangla") or "")
    return generated_id, normalize_space(santali), normalize_space(english), normalize_space(bangla)


def best_match(
    row: dict[str, str],
    gt_rows: list[dict[str, str]],
    english_index: TfIdfIndex,
    english_overlap_index: TokenOverlapIndex,
    bangla_overlap_index: TokenOverlapIndex,
    santali_overlap_index: TokenOverlapIndex,
) -> dict[str, str]:
    generated_id, santali, english, bangla = generated_fields(row)
    ew_scores = english_index.query(english)
    eo_scores = english_overlap_index.query(english)
    bo_scores = bangla_overlap_index.query(bangla)
    so_scores = santali_overlap_index.query(santali)
    candidates = set(ew_scores) | set(eo_scores) | set(bo_scores) | set(so_scores)

    best_id = -1
    best_score = 0.0
    best_scores = (0.0, 0.0, 0.0, 0.0)
    for doc_id in candidates:
        ew = ew_scores.get(doc_id, 0.0)
        eo = eo_scores.get(doc_id, 0.0)
        bo = bo_scores.get(doc_id, 0.0)
        so = so_scores.get(doc_id, 0.0)
        combined = max(ew, eo * 0.95, bo * 0.90, so * 0.50)
        if combined > best_score:
            best_id = doc_id
            best_score = combined
            best_scores = (ew, eo, bo, so)

    gt = gt_rows[best_id] if best_id >= 0 else {}
    ew, eo, bo, so = best_scores
    generated_terms = set(english_terms(english))
    gt_terms = set(english_terms(gt.get("gt_english", "")))
    content_overlap = len(generated_terms & gt_terms)
    content_jaccard = content_overlap / max(len(generated_terms | gt_terms), 1)

    exact_santali = normalize_exact(santali) == normalize_exact(gt.get("gt_santali", ""))
    exact_english = normalize_exact(english) == normalize_exact(gt.get("gt_english", ""))
    exact_bangla = bool(bangla) and normalize_exact(bangla) == normalize_exact(gt.get("gt_bangla", ""))

    return {
        "generated_id": generated_id,
        "generated_santali": santali,
        "generated_english": english,
        "generated_bangla": bangla,
        "best_gt_source": gt.get("gt_source", ""),
        "best_gt_id": gt.get("gt_id", ""),
        "best_gt_santali": gt.get("gt_santali", ""),
        "best_gt_english": gt.get("gt_english", ""),
        "best_gt_bangla": gt.get("gt_bangla", ""),
        "combined_score": f"{best_score:.4f}",
        "english_word_score": f"{ew:.4f}",
        "english_overlap_score": f"{eo:.4f}",
        "bangla_overlap_score": f"{bo:.4f}",
        "santali_overlap_score": f"{so:.4f}",
        "content_word_overlap": str(content_overlap),
        "content_jaccard": f"{content_jaccard:.4f}",
        "exact_santali_match": str(exact_santali).lower(),
        "exact_english_match": str(exact_english).lower(),
        "exact_bangla_match": str(exact_bangla).lower(),
        "strict_reviewable": str(best_score >= 0.75 and content_overlap >= 2 and content_jaccard >= 0.25).lower(),
        "weak_auto_match": str(best_score >= 0.55).lower(),
    }


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * pct)
    return ordered[index]


def summarize_file(path: Path, generated_rows: list[dict[str, str]], strict_errors: int, matches: list[dict[str, str]]) -> dict[str, str]:
    scores = [float(match["combined_score"]) for match in matches]
    ids, santali_values, english_values, bangla_values = [], [], [], []
    for row in generated_rows:
        generated_id, santali, english, bangla = generated_fields(row)
        ids.append(generated_id)
        santali_values.append(santali)
        english_values.append(english)
        bangla_values.append(bangla)

    return {
        "file": path.name,
        "rows": str(len(generated_rows)),
        "strict_json_errors": str(strict_errors),
        "unique_ids": str(len(set(ids))),
        "unique_santali": str(len(set(santali_values))),
        "blank_santali": str(sum(is_blank(value) for value in santali_values)),
        "blank_english": str(sum(is_blank(value) for value in english_values)),
        "blank_bangla": str(sum(is_blank(value) for value in bangla_values)),
        "one_token_santali": str(sum(token_count(value) <= 1 for value in santali_values)),
        "two_or_fewer_token_santali": str(sum(token_count(value) <= 2 for value in santali_values)),
        "english_like_santali": str(sum(english_like_santali(value) for value in santali_values)),
        "mojibake_rows": str(sum(has_mojibake(json.dumps(row, ensure_ascii=False)) for row in generated_rows)),
        "exact_santali_matches": str(sum(match["exact_santali_match"] == "true" for match in matches)),
        "exact_english_matches": str(sum(match["exact_english_match"] == "true" for match in matches)),
        "exact_bangla_matches": str(sum(match["exact_bangla_match"] == "true" for match in matches)),
        "weak_auto_matches_score_ge_0_55": str(sum(match["weak_auto_match"] == "true" for match in matches)),
        "strict_reviewable_score_ge_0_75_content_ge_2": str(sum(match["strict_reviewable"] == "true" for match in matches)),
        "mean_score": f"{statistics.mean(scores):.4f}" if scores else "0.0000",
        "median_score": f"{statistics.median(scores):.4f}" if scores else "0.0000",
        "p90_score": f"{percentile(scores, 0.90):.4f}" if scores else "0.0000",
        "max_score": f"{max(scores):.4f}" if scores else "0.0000",
    }


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    gt_rows = load_ground_truth()
    write_csv(
        GT_COMBINED_CSV,
        gt_rows,
        ["gt_source", "gt_id", "gt_santali", "gt_english", "gt_bangla", "gt_extra"],
    )

    english_index = TfIdfIndex([row["gt_english"] for row in gt_rows], english_terms)
    english_overlap_index = TokenOverlapIndex([row["gt_english"] for row in gt_rows], english_terms)
    bangla_overlap_index = TokenOverlapIndex([row["gt_bangla"] for row in gt_rows], unicode_tokens)
    santali_overlap_index = TokenOverlapIndex([row["gt_santali"] for row in gt_rows], unicode_tokens)

    summary_rows: list[dict[str, str]] = []
    top_matches: list[dict[str, str]] = []
    low_samples: list[dict[str, str]] = []

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

        low = [match for match in matches if float(match["combined_score"]) < 0.30]
        low_samples.extend(low[:100])

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

    print(f"Ground-truth rows: {len(gt_rows):,}")
    print(f"Wrote combined ground truth: {GT_COMBINED_CSV}")
    print(f"Wrote summary: {SUMMARY_CSV}")
    print(f"Wrote top strict matches: {TOP_MATCHES_CSV} ({len(top_matches):,} rows)")
    print(f"Wrote low-score samples: {LOW_SAMPLE_CSV} ({len(low_samples):,} rows)")
    for row in summary_rows:
        print(
            row["file"],
            "rows=", row["rows"],
            "strict=", row["strict_reviewable_score_ge_0_75_content_ge_2"],
            "weak=", row["weak_auto_matches_score_ge_0_55"],
            "mean=", row["mean_score"],
            "blank_en=", row["blank_english"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
