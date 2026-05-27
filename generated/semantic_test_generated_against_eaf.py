from __future__ import annotations

import csv
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median


ROOT = Path(__file__).resolve().parent
GENERATED_CSV = ROOT / "santali_synthetic_40k_sentences.csv"
EAF_UTTERANCE_CSV = ROOT / "initial_data_eaf_utterances.csv"

MATCHES_CSV = ROOT / "generated_vs_eaf_semantic_matches.csv"
SUMMARY_CSV = ROOT / "generated_vs_eaf_semantic_summary.csv"
HIGH_CONFIDENCE_CSV = ROOT / "generated_high_confidence_against_eaf.csv"
LOW_CONFIDENCE_SAMPLE_CSV = ROOT / "generated_low_confidence_sample_against_eaf.csv"

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
}

SANTALI_COLUMNS = ["ref", "mb", "ipa", "ipa_original", "ipa_normalized", "segmentation"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def clean_for_chars(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[\[\]{}()\"'“”‘’.,!?;:।|/\\]+", " ", text)
    return normalize_space(text)


def english_word_terms(text: str) -> list[str]:
    text = clean_for_chars(text)
    raw_terms = re.findall(r"[a-z]+(?:'[a-z]+)?", text)
    terms: list[str] = []
    for term in raw_terms:
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


def unicode_tokens(text: str) -> list[str]:
    text = clean_for_chars(text)
    return [token for token in re.split(r"[\s\-]+", text) if token]


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


def eaf_english(row: dict[str, str]) -> str:
    return normalize_space(row.get("english") or row.get("ft") or "")


def eaf_bangla(row: dict[str, str]) -> str:
    return normalize_space(row.get("bangla") or "")


def eaf_santali(row: dict[str, str]) -> str:
    return normalize_space(" ".join(row.get(column, "") for column in SANTALI_COLUMNS))


def confidence(score: float) -> str:
    if score >= 0.55:
        return "high"
    if score >= 0.30:
        return "medium"
    return "low"


def score_generated_row(
    generated: dict[str, str],
    eaf_rows: list[dict[str, str]],
    english_word_index: TfIdfIndex,
    english_overlap_index: TokenOverlapIndex,
    bangla_overlap_index: TokenOverlapIndex,
    santali_overlap_index: TokenOverlapIndex,
) -> dict[str, str]:
    english_text = generated["english_gloss"]
    bangla_text = generated["bangla_gloss"]
    santali_text = generated["santali_sentence"]

    english_word_scores = english_word_index.query(english_text)
    english_overlap_scores = english_overlap_index.query(english_text)
    bangla_overlap_scores = bangla_overlap_index.query(bangla_text)
    santali_overlap_scores = santali_overlap_index.query(santali_text)

    candidates = (
        set(english_word_scores)
        | set(english_overlap_scores)
        | set(bangla_overlap_scores)
        | set(santali_overlap_scores)
    )

    best_doc_id = -1
    best_combined = -1.0
    best_scores = (0.0, 0.0, 0.0, 0.0)
    for doc_id in candidates:
        ew = english_word_scores.get(doc_id, 0.0)
        eo = english_overlap_scores.get(doc_id, 0.0)
        bo = bangla_overlap_scores.get(doc_id, 0.0)
        so = santali_overlap_scores.get(doc_id, 0.0)
        # A max-style score is intentional: many EAF rows have English only, while
        # the generated data has both English and Bangla glosses.
        combined = max(ew, eo * 0.95, bo * 0.90, so * 0.50)
        if combined > best_combined:
            best_doc_id = doc_id
            best_combined = combined
            best_scores = (ew, eo, bo, so)

    if best_doc_id < 0:
        eaf = {}
        best_combined = 0.0
    else:
        eaf = eaf_rows[best_doc_id]

    ew, ec, bc, sc = best_scores
    return {
        "generated_id": generated["sentence_id"],
        "template_id": generated["template_id"],
        "generated_santali": santali_text,
        "generated_english": english_text,
        "generated_bangla": bangla_text,
        "best_eaf_file": eaf.get("file", ""),
        "best_eaf_utterance_id": eaf.get("utterance_id", ""),
        "best_eaf_santali": eaf_santali(eaf),
        "best_eaf_english": eaf_english(eaf),
        "best_eaf_bangla": eaf_bangla(eaf),
        "combined_score": f"{best_combined:.4f}",
        "english_word_score": f"{ew:.4f}",
        "english_overlap_score": f"{ec:.4f}",
        "bangla_overlap_score": f"{bc:.4f}",
        "santali_overlap_score": f"{sc:.4f}",
        "confidence": confidence(best_combined),
    }


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * pct)))
    return ordered[index]


def make_summary(matches: list[dict[str, str]], generated_rows: list[dict[str, str]], eaf_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    scores = [float(row["combined_score"]) for row in matches]
    confidence_counts = Counter(row["confidence"] for row in matches)
    template_counts = Counter(row["template_id"] for row in matches)
    high_by_template = Counter(row["template_id"] for row in matches if row["confidence"] == "high")
    eaf_hit_count = len({(row["best_eaf_file"], row["best_eaf_utterance_id"]) for row in matches if row["best_eaf_file"]})

    summary = [
        metric("generated_rows", len(generated_rows), "PASS", "Generated rows tested."),
        metric("eaf_ground_truth_rows", len(eaf_rows), "PASS", "EAF utterance rows used as ground truth."),
        metric("mean_combined_similarity", f"{mean(scores):.4f}", "INFO", "Mean nearest-neighbor score."),
        metric("median_combined_similarity", f"{median(scores):.4f}", "INFO", "Median nearest-neighbor score."),
        metric("p90_combined_similarity", f"{percentile(scores, 0.90):.4f}", "INFO", "90th percentile score."),
        metric("p95_combined_similarity", f"{percentile(scores, 0.95):.4f}", "INFO", "95th percentile score."),
        metric("max_combined_similarity", f"{max(scores):.4f}", "INFO", "Best score found."),
        metric("high_confidence_rows", confidence_counts["high"], "INFO", "Rows scoring >= 0.55."),
        metric("medium_confidence_rows", confidence_counts["medium"], "INFO", "Rows scoring >= 0.30 and < 0.55."),
        metric("low_confidence_rows", confidence_counts["low"], "INFO", "Rows scoring < 0.30."),
        metric("eaf_rows_used_as_best_match", eaf_hit_count, "INFO", "Distinct EAF rows selected by at least one generated row."),
        metric("top_generated_templates", "; ".join(f"{k}={v}" for k, v in template_counts.most_common(8)), "INFO", "Template distribution in generated data."),
        metric("top_high_confidence_templates", "; ".join(f"{k}={v}" for k, v in high_by_template.most_common(8)), "INFO", "Templates most often close to EAF ground truth."),
    ]
    return summary


def metric(name: str, value: object, status: str, detail: str) -> dict[str, str]:
    return {"metric": name, "value": str(value), "status": status, "detail": detail}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if not GENERATED_CSV.exists():
        raise FileNotFoundError(GENERATED_CSV)
    if not EAF_UTTERANCE_CSV.exists():
        raise FileNotFoundError(EAF_UTTERANCE_CSV)

    generated_rows = read_csv(GENERATED_CSV)
    eaf_rows = read_csv(EAF_UTTERANCE_CSV)

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

    match_fields = [
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

    write_csv(MATCHES_CSV, matches, match_fields)
    write_csv(
        SUMMARY_CSV,
        make_summary(matches, generated_rows, eaf_rows),
        ["metric", "value", "status", "detail"],
    )

    high_confidence = [row for row in matches if row["confidence"] == "high"]
    low_confidence = [row for row in matches if row["confidence"] == "low"][:500]
    write_csv(HIGH_CONFIDENCE_CSV, high_confidence, match_fields)
    write_csv(LOW_CONFIDENCE_SAMPLE_CSV, low_confidence, match_fields)

    print(f"Wrote semantic matches: {MATCHES_CSV} ({len(matches):,} rows)")
    print(f"Wrote summary: {SUMMARY_CSV}")
    print(f"Wrote high-confidence matches: {HIGH_CONFIDENCE_CSV} ({len(high_confidence):,} rows)")
    print(f"Wrote low-confidence sample: {LOW_CONFIDENCE_SAMPLE_CSV} ({len(low_confidence):,} rows)")
    for row in make_summary(matches, generated_rows, eaf_rows)[:10]:
        print(f"{row['status']}: {row['metric']} = {row['value']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
