from __future__ import annotations

import csv
import random
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parent
DICTIONARY_XLSX = ROOT / "MTZ_Santali_Dictionary.xlsx"
REPORT_TXT = ROOT / "Copy of Copy of Final Report.Team Clausetrophobic (1).txt"
OUTPUT_CSV = ROOT / "santali_synthetic_40k_sentences.csv"

TARGET_ROWS = 40_000
RANDOM_SEED = 20260526

NS = {
    "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "pkgrel": "http://schemas.openxmlformats.org/package/2006/relationships",
}

SUBJECTS = [
    {"san": "iɲ", "en": "I", "bn": "আমি", "neg": "bɔ-iɲ", "cop": "", "poss": "iɲ-aʔ", "be": "am", "do": "do"},
    {"san": "am", "en": "you", "bn": "তুমি", "neg": "ba-m", "cop": "-m", "poss": "am-aʔ", "be": "are", "do": "do"},
    {"san": "uni", "en": "he/she", "bn": "সে", "neg": "ba-e", "cop": "-e", "poss": "uni-aʔ", "be": "is", "do": "does"},
    {"san": "abo", "en": "we", "bn": "আমরা", "neg": "ba-bon", "cop": "-bon", "poss": "abo-aʔ", "be": "are", "do": "do"},
]

DEMONSTRATIVES = [
    ("nɔa", "this", "এটা"),
    ("ona", "that", "ওটা"),
]

SOURCES = {
    "POSS_EXIST": "Report 5.1.17 possessive existential",
    "DEMO_NOM": "Report 5.1.1/5.1.17 demonstrative nominal predicate",
    "DEMO_NEG_NOM": "Report 5.1.15 nominal predicate negation",
    "DEMO_ADJ": "Report 5.1.14 adjectival predicate",
    "DEMO_NEG_ADJ": "Report 5.1.14 adjectival negation",
    "SUBJ_ADJ": "Report 5.1.14 subject adjectival predicate",
    "SUBJ_NEG_ADJ": "Report 5.1.14 negative adjectival predicate",
    "SUBJ_VERB_PRESENT": "Report 5.1.11 present verbal predicate",
    "SUBJ_VERB_NEG_PRESENT": "Report 5.1.11 present verbal negation",
    "SUBJ_VERB_PAST": "Report 5.1.12 past verbal predicate",
    "SUBJ_VERB_FUTURE": "Report 5.1.3/5.1.6 future verbal predicate",
    "SUBJ_VERB_NEG_FUTURE": "Report 5.1.6 future verbal negation",
    "OBJECT_VERB": "Report 5.1.11 object plus verb predicate",
    "OBJECT_VERB_NEG": "Report 5.1.11 negated object plus verb predicate",
    "IMPERATIVE": "Report 5.1.8 imperative",
    "PROHIBITIVE": "Report 5.1.8 prohibitive",
    "LOC_EXIST": "Report 5.1.10/5.1.13 locative existential",
    "LOC_NEG_EXIST": "Report 5.1.10 negative existential",
    "YESNO_VERB": "Report 5.1.7 interrogative verbal predicate",
    "YESNO_ADJ": "Report 5.1.7 adjectival yes/no question",
    "POSS_NOM": "Report 5.1.17 possessive nominal predicate",
    "ADJ_NOUN": "Report 5.1.17 adjective plus noun predicate",
    "NUM_NOUN_EXIST": "Report 5.1.17 number plus noun existential",
    "ADV_VERB": "Report 5.1.13 adverbial verbal predicate",
    "PLACE_VERB": "Report 5.1.13 locative verbal predicate",
}


def parse_xml(zf: ZipFile, name: str) -> ET.Element:
    return ET.fromstring(zf.read(name))


def column_index(cell_ref: str) -> int:
    letters = "".join(ch for ch in cell_ref if ch.isalpha())
    index = 0
    for ch in letters:
        index = index * 26 + (ord(ch.upper()) - ord("A") + 1)
    return index - 1


def cell_text(cell: ET.Element, shared_strings: list[str]) -> str:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return "".join(t.text or "" for t in cell.findall(".//main:t", NS)).strip()

    value = cell.find("main:v", NS)
    if value is None or value.text is None:
        return ""
    if cell_type == "s":
        return shared_strings[int(value.text)].strip()
    return value.text.strip()


def row_values(row: ET.Element, shared_strings: list[str]) -> list[str]:
    values: list[str] = []
    last_index = -1
    for cell in row.findall("main:c", NS):
        ref = cell.attrib.get("r", "")
        index = column_index(ref) if ref else last_index + 1
        while len(values) < index:
            values.append("")
        values.append(cell_text(cell, shared_strings))
        last_index = index
    return values


def dictionary_sheet_path(zf: ZipFile) -> str:
    workbook = parse_xml(zf, "xl/workbook.xml")
    rels = parse_xml(zf, "xl/_rels/workbook.xml.rels")
    rid_to_target = {
        rel.attrib["Id"]: rel.attrib["Target"]
        for rel in rels.findall("pkgrel:Relationship", NS)
    }

    for sheet in workbook.findall(".//main:sheet", NS):
        if sheet.attrib.get("name") == "Dictionary":
            rid = sheet.attrib[
                "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
            ]
            target = rid_to_target[rid]
            return "xl/" + target.lstrip("/") if not target.startswith("xl/") else target
    raise RuntimeError("Could not find a sheet named 'Dictionary'.")


def read_dictionary(path: Path) -> list[dict[str, str]]:
    with ZipFile(path) as zf:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = parse_xml(zf, "xl/sharedStrings.xml")
            for item in root.findall("main:si", NS):
                shared_strings.append(
                    "".join(t.text or "" for t in item.findall(".//main:t", NS))
                )

        sheet = parse_xml(zf, dictionary_sheet_path(zf))
        rows = sheet.findall(".//main:sheetData/main:row", NS)
        headers = row_values(rows[0], shared_strings)
        entries: list[dict[str, str]] = []
        for row in rows[1:]:
            values = row_values(row, shared_strings)
            values += [""] * (len(headers) - len(values))
            record = dict(zip(headers, values))
            entries.append(
                {
                    "hw": record.get("Santali (Headword)", "").strip(),
                    "ipa": record.get("IPA", "").strip(),
                    "pos": record.get("POS", "").strip(),
                    "en": record.get("English", "").strip(),
                    "bn": record.get("Bangla", "").strip(),
                    "domain": record.get("Semantic Domain", "").strip(),
                }
            )
        return entries


def pos_tokens(pos: str) -> set[str]:
    normalized = pos.lower().replace(" ", "")
    return {
        token.strip(".")
        for token in re.split(r"[/,;]+", normalized)
        if token.strip(".")
    }


def usable_headword(headword: str) -> bool:
    if not headword or headword in {"-", "–", "—"}:
        return False
    if headword.startswith("-") or headword.endswith("-"):
        return False
    return len(headword) <= 42


def classify(entries: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    pools = {key: [] for key in ["n", "v", "adj", "adv", "num", "place"]}
    seen = {key: set() for key in pools}

    for entry in entries:
        if not usable_headword(entry["hw"]):
            continue
        tokens = pos_tokens(entry["pos"])
        for key in ["n", "v", "adj", "adv", "num"]:
            marker = (entry["hw"], entry["en"])
            if key in tokens and marker not in seen[key]:
                pools[key].append(entry)
                seen[key].add(marker)

        marker = (entry["hw"], entry["en"])
        if (
            "n" in tokens
            and entry["domain"].lower() in {"place", "nature"}
            and marker not in seen["place"]
        ):
            pools["place"].append(entry)
            seen["place"].add(marker)
    return pools


def english(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip() or "item"
    return text[:1].lower() + text[1:]


def bangla(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip() or "বস্তু"


def add_suffix(word: str, suffix: str) -> str:
    return word if word.endswith(suffix) else word + suffix


def slot(entry: dict[str, str] | None) -> str:
    if not entry:
        return ""
    return f"{entry['hw']} [{entry['pos']}; {entry['domain']}; {entry['en']}; {entry['bn']}]"


def make_row(
    template_id: str,
    sentence_type: str,
    santali: str,
    english_gloss: str,
    bangla_gloss: str,
    *,
    subject: dict[str, str] | None = None,
    noun: dict[str, str] | None = None,
    verb: dict[str, str] | None = None,
    adjective: dict[str, str] | None = None,
    adverb: dict[str, str] | None = None,
    place: dict[str, str] | None = None,
    number: dict[str, str] | None = None,
) -> dict[str, str]:
    return {
        "sentence_id": "",
        "santali_sentence": santali,
        "english_gloss": english_gloss,
        "bangla_gloss": bangla_gloss,
        "template_id": template_id,
        "sentence_type": sentence_type,
        "source_pattern": SOURCES[template_id],
        "subject_slot": subject["san"] if subject else "",
        "noun_slot": slot(noun),
        "verb_slot": slot(verb),
        "adjective_slot": slot(adjective),
        "adverb_slot": slot(adverb),
        "place_slot": slot(place),
        "number_slot": slot(number),
        "synthetic": "true",
        "quality_note": "synthetic; template-based; needs native-speaker validation",
        "source_files": f"{REPORT_TXT.name}; {DICTIONARY_XLSX.name}",
    }


def generate_one(rng: random.Random, pools: dict[str, list[dict[str, str]]]) -> dict[str, str]:
    template_id = rng.choice(list(SOURCES))
    subject = rng.choice(SUBJECTS)
    demo_san, demo_en, demo_bn = rng.choice(DEMONSTRATIVES)
    noun = rng.choice(pools["n"])
    verb = rng.choice(pools["v"])
    adjective = rng.choice(pools["adj"])
    adverb = rng.choice(pools["adv"])
    place = rng.choice(pools["place"])
    number = rng.choice(pools["num"])

    if template_id == "POSS_EXIST":
        return make_row(
            template_id,
            "possessive/existential",
            f"{subject['poss']} mit't'e {noun['hw']} minaʔ",
            f"{subject['en']} have one {english(noun['en'])}.",
            f"{subject['bn']} একটি {bangla(noun['bn'])} আছে।",
            subject=subject,
            noun=noun,
        )
    if template_id == "DEMO_NOM":
        return make_row(
            template_id,
            "nominal predicate",
            f"{demo_san} dɔ {noun['hw']} kan-a",
            f"{demo_en.capitalize()} is {english(noun['en'])}.",
            f"{demo_bn} {bangla(noun['bn'])}।",
            noun=noun,
        )
    if template_id == "DEMO_NEG_NOM":
        return make_row(
            template_id,
            "negative nominal predicate",
            f"{demo_san} dɔ {noun['hw']} dɔ ba-ŋ kan-a",
            f"{demo_en.capitalize()} is not {english(noun['en'])}.",
            f"{demo_bn} {bangla(noun['bn'])} নয়।",
            noun=noun,
        )
    if template_id == "DEMO_ADJ":
        return make_row(
            template_id,
            "adjectival predicate",
            f"{demo_san} dɔ {adjective['hw']} kan-a",
            f"{demo_en.capitalize()} is {english(adjective['en'])}.",
            f"{demo_bn} {bangla(adjective['bn'])}।",
            adjective=adjective,
        )
    if template_id == "DEMO_NEG_ADJ":
        return make_row(
            template_id,
            "negative adjectival predicate",
            f"{demo_san} dɔ ba-ŋ {adjective['hw']} kan-a",
            f"{demo_en.capitalize()} is not {english(adjective['en'])}.",
            f"{demo_bn} {bangla(adjective['bn'])} নয়।",
            adjective=adjective,
        )
    if template_id == "SUBJ_ADJ":
        return make_row(
            template_id,
            "subject adjectival predicate",
            f"{subject['san']} dɔ {adjective['hw']} kan-a{subject['cop']}",
            f"{subject['en']} {subject['be']} {english(adjective['en'])}.",
            f"{subject['bn']} {bangla(adjective['bn'])}।",
            subject=subject,
            adjective=adjective,
        )
    if template_id == "SUBJ_NEG_ADJ":
        return make_row(
            template_id,
            "subject negative adjectival predicate",
            f"{subject['san']} dɔ {subject['neg']} {adjective['hw']} kan-a",
            f"{subject['en']} {subject['be']} not {english(adjective['en'])}.",
            f"{subject['bn']} {bangla(adjective['bn'])} নয়।",
            subject=subject,
            adjective=adjective,
        )
    if template_id == "SUBJ_VERB_PRESENT":
        return make_row(
            template_id,
            "present verbal predicate",
            f"{subject['san']} dɔ {verb['hw']} kan-a{subject['cop']}",
            f"{subject['en']} {subject['do']} {english(verb['en'])}.",
            f"{subject['bn']} {bangla(verb['bn'])}।",
            subject=subject,
            verb=verb,
        )
    if template_id == "SUBJ_VERB_NEG_PRESENT":
        return make_row(
            template_id,
            "negative present verbal predicate",
            f"{subject['san']} dɔ {subject['neg']} {verb['hw']} kan-a",
            f"{subject['en']} {subject['do']} not {english(verb['en'])}.",
            f"{subject['bn']} {bangla(verb['bn'])} না।",
            subject=subject,
            verb=verb,
        )
    if template_id == "SUBJ_VERB_PAST":
        return make_row(
            template_id,
            "past verbal predicate",
            f"{subject['san']} dɔ {add_suffix(verb['hw'], '-len-a')}",
            f"{subject['en']} did {english(verb['en'])}.",
            f"{subject['bn']} {bangla(verb['bn'])} করেছিল।",
            subject=subject,
            verb=verb,
        )
    if template_id == "SUBJ_VERB_FUTURE":
        return make_row(
            template_id,
            "future verbal predicate",
            f"{subject['san']} dɔ {add_suffix(verb['hw'], '-a')}",
            f"{subject['en']} will {english(verb['en'])}.",
            f"{subject['bn']} {bangla(verb['bn'])} করবে।",
            subject=subject,
            verb=verb,
        )
    if template_id == "SUBJ_VERB_NEG_FUTURE":
        return make_row(
            template_id,
            "negative future verbal predicate",
            f"{subject['san']} dɔ {subject['neg']} {add_suffix(verb['hw'], '-a')}",
            f"{subject['en']} will not {english(verb['en'])}.",
            f"{subject['bn']} {bangla(verb['bn'])} করবে না।",
            subject=subject,
            verb=verb,
        )
    if template_id == "OBJECT_VERB":
        return make_row(
            template_id,
            "object verbal predicate",
            f"{subject['san']} dɔ {noun['hw']}-i {add_suffix(verb['hw'], '-a')}",
            f"{subject['en']} {subject['do']} {english(verb['en'])} with {english(noun['en'])}.",
            f"{subject['bn']} {bangla(noun['bn'])} {bangla(verb['bn'])}।",
            subject=subject,
            noun=noun,
            verb=verb,
        )
    if template_id == "OBJECT_VERB_NEG":
        return make_row(
            template_id,
            "negative object verbal predicate",
            f"{subject['san']} dɔ {noun['hw']}-to {subject['neg']} {add_suffix(verb['hw'], '-a')}",
            f"{subject['en']} {subject['do']} not {english(verb['en'])} with {english(noun['en'])}.",
            f"{subject['bn']} {bangla(noun['bn'])} {bangla(verb['bn'])} না।",
            subject=subject,
            noun=noun,
            verb=verb,
        )
    if template_id == "IMPERATIVE":
        return make_row(
            template_id,
            "imperative",
            f"{verb['hw']} me",
            f"Please {english(verb['en'])}.",
            f"{bangla(verb['bn'])}।",
            verb=verb,
        )
    if template_id == "PROHIBITIVE":
        return make_row(
            template_id,
            "negative imperative/prohibitive",
            f"alɔ-m {add_suffix(verb['hw'], '-a')}",
            f"Do not {english(verb['en'])}.",
            f"{bangla(verb['bn'])} না।",
            verb=verb,
        )
    if template_id == "LOC_EXIST":
        return make_row(
            template_id,
            "locative existential",
            f"{place['hw']}-re {noun['hw']} minaʔ",
            f"There is {english(noun['en'])} in/at {english(place['en'])}.",
            f"{bangla(place['bn'])}-তে {bangla(noun['bn'])} আছে।",
            noun=noun,
            place=place,
        )
    if template_id == "LOC_NEG_EXIST":
        return make_row(
            template_id,
            "negative locative existential",
            f"{place['hw']}-re {noun['hw']} banuʔ-a",
            f"There is no {english(noun['en'])} in/at {english(place['en'])}.",
            f"{bangla(place['bn'])}-তে {bangla(noun['bn'])} নেই।",
            noun=noun,
            place=place,
        )
    if template_id == "YESNO_VERB":
        question_do = "Does" if subject["do"] == "does" else "Do"
        return make_row(
            template_id,
            "interrogative verbal predicate",
            f"{subject['san']} dɔ {verb['hw']} kan-a?",
            f"{question_do} {subject['en']} {english(verb['en'])}?",
            f"{subject['bn']} কি {bangla(verb['bn'])}?",
            subject=subject,
            verb=verb,
        )
    if template_id == "YESNO_ADJ":
        return make_row(
            template_id,
            "interrogative adjectival predicate",
            f"{demo_san} dɔ {adjective['hw']} kan-a?",
            f"Is {demo_en} {english(adjective['en'])}?",
            f"{demo_bn} কি {bangla(adjective['bn'])}?",
            adjective=adjective,
        )
    if template_id == "POSS_NOM":
        return make_row(
            template_id,
            "possessive nominal predicate",
            f"{demo_san} dɔ {subject['poss']} {noun['hw']} kan-a",
            f"{demo_en.capitalize()} is {subject['en']}'s {english(noun['en'])}.",
            f"{demo_bn} {subject['bn']}-এর {bangla(noun['bn'])}।",
            subject=subject,
            noun=noun,
        )
    if template_id == "ADJ_NOUN":
        return make_row(
            template_id,
            "adjective plus noun predicate",
            f"{demo_san} dɔ {adjective['hw']} {noun['hw']} kan-a",
            f"{demo_en.capitalize()} is a {english(adjective['en'])} {english(noun['en'])}.",
            f"{demo_bn} {bangla(adjective['bn'])} {bangla(noun['bn'])}।",
            noun=noun,
            adjective=adjective,
        )
    if template_id == "NUM_NOUN_EXIST":
        return make_row(
            template_id,
            "numbered existential",
            f"{number['hw']} {noun['hw']} minaʔ",
            f"There are/is {english(number['en'])} {english(noun['en'])}.",
            f"{bangla(number['bn'])} {bangla(noun['bn'])} আছে।",
            noun=noun,
            number=number,
        )
    if template_id == "ADV_VERB":
        return make_row(
            template_id,
            "adverbial verbal predicate",
            f"{subject['san']} dɔ {adverb['hw']} {verb['hw']} kan-a",
            f"{subject['en']} {subject['do']} {english(verb['en'])} {english(adverb['en'])}.",
            f"{subject['bn']} {bangla(adverb['bn'])} {bangla(verb['bn'])}।",
            subject=subject,
            verb=verb,
            adverb=adverb,
        )

    return make_row(
        "PLACE_VERB",
        "locative verbal predicate",
        f"{subject['san']} dɔ {place['hw']}-re {verb['hw']} kan-a",
        f"{subject['en']} {subject['do']} {english(verb['en'])} in/at {english(place['en'])}.",
        f"{subject['bn']} {bangla(place['bn'])}-তে {bangla(verb['bn'])}।",
        subject=subject,
        verb=verb,
        place=place,
    )


def build_rows(pools: dict[str, list[dict[str, str]]]) -> list[dict[str, str]]:
    rng = random.Random(RANDOM_SEED)
    rows: list[dict[str, str]] = []
    seen_sentences: set[str] = set()
    attempts = 0
    max_attempts = TARGET_ROWS * 100

    while len(rows) < TARGET_ROWS and attempts < max_attempts:
        attempts += 1
        row = generate_one(rng, pools)
        sentence = row["santali_sentence"]
        if sentence in seen_sentences:
            continue
        seen_sentences.add(sentence)
        row["sentence_id"] = f"SYN-{len(rows) + 1:05d}"
        rows.append(row)

    if len(rows) != TARGET_ROWS:
        raise RuntimeError(f"Generated {len(rows)} rows, expected {TARGET_ROWS}.")
    return rows


def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    fieldnames = [
        "sentence_id",
        "santali_sentence",
        "english_gloss",
        "bangla_gloss",
        "template_id",
        "sentence_type",
        "source_pattern",
        "subject_slot",
        "noun_slot",
        "verb_slot",
        "adjective_slot",
        "adverb_slot",
        "place_slot",
        "number_slot",
        "synthetic",
        "quality_note",
        "source_files",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if not DICTIONARY_XLSX.exists():
        raise FileNotFoundError(DICTIONARY_XLSX)
    if not REPORT_TXT.exists():
        raise FileNotFoundError(REPORT_TXT)

    entries = read_dictionary(DICTIONARY_XLSX)
    pools = classify(entries)
    for key in ["n", "v", "adj", "adv", "num", "place"]:
        if not pools[key]:
            raise RuntimeError(f"No usable entries found for {key!r}.")

    rows = build_rows(pools)
    write_csv(rows, OUTPUT_CSV)

    print(f"Wrote {len(rows):,} rows to {OUTPUT_CSV}")
    print(", ".join(f"{key}={len(pools[key]):,}" for key in ["n", "v", "adj", "adv", "num", "place"]))
    print(f"Unique Santali sentences: {len({row['santali_sentence'] for row in rows}):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
