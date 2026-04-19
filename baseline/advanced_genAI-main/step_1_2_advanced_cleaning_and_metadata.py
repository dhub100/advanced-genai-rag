#!/usr/bin/env python3
"""
Script: step_1_2_advanced_cleaning_and_metadata.py

Purpose:
  Read JSON from step_1_hybrid.py (which has raw_text, paragraphs, minimal fields),
  then perform advanced cleanup and enrichment:
    1) Remove textual boilerplate disclaimers/footers (via regex patterns).
    2) Build a frequency dictionary of paragraphs across all docs and remove repeated paragraphs
       exceeding a threshold (default=5).
    3) Perform advanced metadata extraction:
       - Language detection (Lingua)
       - (Optional) Date extraction from path or text
       - Named Entity Recognition (spaCy)
       - Keyword Extraction (YAKE)
       - Summaries (simple bullet or first-two-sentence approach)
    4) Write final JSON with fields like:
        {
         "doc_id": "9b7f034dc9cd5fd1b...",
         "filename": "example.html",
         "domain": "ethz.ch",
         "language": "de",
         "title": "",                           --> not implemented
         "date": "2023-05-01",                  --> always YYYY-MM-01
         "year": 2023,
         "month": 5,
         "source": "ETH News",

         "main_content": "Margrit Leuthold has been the Executive ...",
         "paragraphs_original": [
            "## About the author",
                "Margrit Leuthold has been the Executive ...",
                "## Subscribe to Newsletter",
                "Subscribe to the Newsletter for internal news",
                "## Staffnet",
                "Info portal for employees ..."
        ],
         "paragraphs_cleaned": [
           "Margrit Leuthold has been the Executive ...",
           "... cleaned paragraph 2 if any ..."
         ],

         "named_entities": [
           {"text": "Margrit Leuthold", "label": "PERSON"},
           {"text": "ETH Zurich", "label": "ORG"}
         ],
         "keywords": ["Margrit Leuthold", "Bangalore", "Executive Director"],
         "summary": "Margrit Leuthold has been the Executive Director ...",
         "text_stats": {
           "char_count": 864,
            "word_count": 128,
            "paragraph_count": 1
         },
         "semantic_chunk_hints": [
           {"type": "paragraph_boundaries", "count": 1}
         ],
         "embedding_vector": [],
         "doc_embedding": []
       }

Example usage:
    python step_2_advanced_cleaning.py data_cleaned/minimal_hybrid data_cleaned/advanced --threshold 5
"""

import os
import re
import json
import time
import logging
import argparse
from pathlib import Path
import dateparser

import spacy
import yake
from lingua import Language, LanguageDetectorBuilder

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Build a Lingua detector for 4 languages
detector = (
    LanguageDetectorBuilder
    .from_languages(Language.ENGLISH, Language.GERMAN, Language.FRENCH, Language.ITALIAN)
    .with_preloaded_language_models()
    .build()
)

# Lazy-loaded spaCy models by language code
spacy_models = {}
# Lazy-loaded YAKE extractors
yake_extractors = {}

# Regex patterns that match disclaimer/footer lines to remove
BOILERPLATE_PATTERNS = [
    # ↓↓↓ ACTIVE PATTERNS (e.g., ONLY for complete lines) ↓↓↓

    # e.g. a line containing only "Staffnet"
    r"^Staffnet\s*$",

    # a line containing only "Newsletter abonnieren"
    r"^Newsletter\sabonnieren\s*$",

    # a line containing only "call_made"
    r"^call_made\s*$",

    # a line containing only "externe Seite"
    r"^externe\sSeite\s*$",

    # a line containing only "vertical_align_bottom"
    r"^vertical_align_bottom\s*$",

    # a line containing only "Download"
    r"^Download\s*$",


    # ↓↓↓ EVERYTHING ELSE IS COMMENTED OUT BELOW ↓↓↓

    # r"\(PDF.*\)",
    # r"\.pdf",
    # r"\d+\s?MB",
    # r"In\sKürze",
    #
    # # disclaimers
    # r"Info-Portal",
    # r"Immer\saktuell\sinformiert",
    # r"Möchten\sSie\sstets.*",
    # r"subscribe.*newsletter",
    # r"^Tel\.\s*\+?",
    # r"^Kontakt[e]?$",
    # r"Für\sNewsletter.*aktuell.*anmelden",
    # r"\(die Veranstaltung findet in Englisch statt\)",
    # r"^Kontaktadresse",
    # r"^Fax\.\s*\+?",
    # r"^Media\sRelations$",
    # r"Impressum",
    # r"^Anmeldung.*erforderlich.*",
    # r"^Programmwebseite$",
    # r"^Archiv$",
    # r"^Aktuelles$",
    # r"^Login$",
    # r"^Logout$",
    # r"Mehr\sInformation(en)?",
    # r"Keinen\sBeitrag\sverpassen",
    # r"^Disclaimer$",
    # r"^Haftungsausschluss$",
    # r"zurück\s(zur)?\sübersicht",
    # r"^printversion$",
    # r"^Datum:\s*\d{1,2}\.\d{1,2}\.\d{2,4}",
    # r"©\s?\d{4}\s*ETH",
    # r"^Seite\s?drucken$",
    # r"^Teilen$",
    # r"^Zur\sStartseite$",
    # r"^Dokument\serstellen$",
    # r"^Link\skopieren$",
    # r"Folgen\sSie\suns\sauf",
    # r"^Zum\sSeitenanfang$",
    # r"^Zur\sVorlesung\sanmelden$",
    # r"^Jetzt\sregistrieren$",
    # r"^Sharing\sis\scaring$",
    # r"Dieser\sArtikel\sist\seine\sÜbersetzung",
    # r"Follow\sus\son\sTwitter",
    # r"Kontakt\sprüfen\sSie\smit\suns",
    # r"^Mobile\sVersion$",
    # r"^Seitenumbruch$",
    # r"^Zum\sInhaltsverzeichnis$",
    # r"^Datenschutz.*",
    # r"^Legal\snotice$",
    # r"^©\sETH\s.*",
    # r"^Kontaktformular$",
    # r"^Seitenende$",
    # r"^Weiter\szu.*",
    # r"^lesen\sSie\smehr$",
    # r"^Inhalt\sverbergen$",
    # r"^Seiteninhalt$",
    # r"^Footernavigation$",
    # r"^Top\sNavigation$",
    # r"^Sprache\swechseln$",
    # r"^Datenschutz\S*$",
    # r"^Privacy\sPolicy$",
    # r"^Bitte\sklicken\sSie\shier.*",
    # r"^Zum\sNewsletter\sanmelden$",
    # r"^Newsletter\sbestellen$",
    # r"^Sitemap$",
    # r"^Kontakt\s/\s.*",
    # r"^Notfall\s?nummern?$",
    # r"^Support\sKontakt.*",
    # r"^Diesen\sBeitrag\steilen$"
]

def get_spacy_model(lang_code: str):
    """
    Lazy-load spaCy for 'en' or 'de'.
    """
    model_map = {"en": "en_core_web_sm", "de": "de_core_news_sm"}
    if lang_code not in model_map:
        return None
    if lang_code in spacy_models:
        return spacy_models[lang_code]
    try:
        nlp = spacy.load(model_map[lang_code])
        spacy_models[lang_code] = nlp
        return nlp
    except Exception as e:
        logging.error(f"SpaCy load error for {lang_code}: {e}")
        return None

def detect_lang(text: str) -> str:
    """
    Use Lingua to detect language. Return ISO639-1 code like 'en', 'de', etc.
    """
    if not text.strip():
        return ""
    try:
        lang = detector.detect_language_of(text)
        return lang.iso_code_639_1.name.lower() if lang else ""
    except Exception as e:
        logging.error(f"Language detection error: {e}")
        return ""

def parse_date_from_path(file_path: Path):
    """
    Attempt to parse date info from folder structure (YYYY/MM).
    Returns iso_date, year_int, month_int or (None, None, None).
    """
    year = None
    month = None
    # e.g. if path has segments: "2023/05"
    for part in file_path.parts:
        if re.fullmatch(r"\d{4}", part):
            year = part
        elif re.fullmatch(r"(0[1-9]|1[0-2])", part):
            if year:
                month = part.zfill(2)

    if year:
        iso_date = f"{year}-{month or '01'}-01"
        return iso_date, int(year), int(month) if month else 1
    return "", None, None

def extract_entities_spacy(text: str, lang_code: str):
    """
    Named Entity Recognition with spaCy if model is available.
    """
    nlp = get_spacy_model(lang_code)
    if not nlp:
        return []
    doc = nlp(text)
    seen = set()
    ents = []
    for e in doc.ents:
        if e.text not in seen:
            seen.add(e.text)
            ents.append({"text": e.text, "label": e.label_})
    return ents

def extract_keywords_yake(text: str, lang_code: str, top_k=10):
    """
    YAKE keyword extraction for en/de/fr/it. Fallback = en.
    """
    if not text.strip():
        return []
    # pick best guess
    yake_lang = lang_code if lang_code in ["en","de","fr","it"] else "en"

    if yake_lang not in yake_extractors:
        try:
            yake_extractors[yake_lang] = yake.KeywordExtractor(lan=yake_lang, n=3, top=top_k)
        except Exception as e:
            logging.error(f"Failed to init YAKE {yake_lang}: {e}")
            yake_extractors[yake_lang] = yake.KeywordExtractor(lan="en", n=3, top=top_k)

    extractor = yake_extractors[yake_lang]
    try:
        kw_sc = extractor.extract_keywords(text)
        # sort by ascending score
        kw_sc.sort(key=lambda x: x[1])
        return [k for k,_ in kw_sc[:top_k]]
    except Exception as e:
        logging.error(f"YAKE extract error: {e}")
        return []

def simple_summary(text: str, lang_code: str, max_bullets=3):
    """
    1) If bullet lines exist, return first n bullet lines.
    2) Else attempt spaCy sentence splitting, returning first 1-2 sentences.
    3) Else naive split on punctuation.
    """
    lines = text.splitlines()
    bullet_lines = [ln.strip() for ln in lines if ln.strip().startswith("-")]
    if bullet_lines:
        return "\n".join(bullet_lines[:max_bullets])

    # spaCy approach
    nlp = get_spacy_model(lang_code)
    if nlp:
        doc = nlp(text)
        sents = [s.text.strip() for s in doc.sents if s.text.strip()]
        if len(sents) >= 2:
            return sents[0] + " " + sents[1]
        elif sents:
            return sents[0]
        return ""
    else:
        # naive approach
        parts = re.split(r"(?<=[.!?]) +", text)
        if len(parts) >= 2:
            return parts[0].strip() + " " + parts[1].strip()
        elif parts:
            return parts[0].strip()
        return ""

def remove_boilerplate_in_paragraphs(paragraphs: list[str]) -> list[str]:
    """
    Remove lines matching disclaimers or repeated boilerplate from each paragraph.
    Returns updated list of paragraphs without those lines.
    """
    combined_regex = re.compile("|".join(BOILERPLATE_PATTERNS), re.IGNORECASE)
    cleaned_paras = []
    for para in paragraphs:
        # Split paragraph by lines
        lines = para.splitlines()
        # Filter out any lines that match BOILERPLATE_PATTERNS
        filtered_lines = [
            ln for ln in lines
            if not combined_regex.search(ln.strip())
        ]
        # Recombine into a single paragraph
        new_para = " ".join(fl.strip() for fl in filtered_lines if fl.strip())
        if new_para:
            cleaned_paras.append(new_para)
    return cleaned_paras

def main():
    parser = argparse.ArgumentParser(description="Step 2: remove repeated paragraphs, disclaimers, add advanced metadata.")
    parser.add_argument("input_dir", help="Directory with JSON from step_1_hybrid.")
    parser.add_argument("output_dir", help="Directory for final structured JSON.")
    parser.add_argument("--threshold", type=int, default=5,
                        help="Remove paragraphs repeated >= threshold times.")
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    # 1) Gather documents & remove textual disclaimers from paragraphs
    docs = []
    paragraph_counts = {}

    json_files = list(in_dir.rglob("*.json"))
    for js_path in json_files:
        try:
            data = json.loads(js_path.read_text(encoding="utf-8"))
        except Exception as e:
            logging.error(f"Error reading {js_path}: {e}")
            continue

        original_paras = data.get("paragraphs", [])

        # Step 1.1: Remove disclaimers/footers from paragraphs
        refined_paras = remove_boilerplate_in_paragraphs(original_paras)

        # We'll keep track of these refined paras in memory
        docs.append((js_path, data, refined_paras))
        # For building the frequency dictionary
        for p in refined_paras:
            paragraph_counts[p] = paragraph_counts.get(p, 0) + 1

    # 2) Identify repeated paragraphs above threshold
    threshold = args.threshold
    logging.info(f"Repeat paragraph threshold set to {threshold}.")

    # 3) For each doc, remove repeated paragraphs, extract metadata
    out_count = 0
    for js_path, data, refined_paras in docs:
        # Filter out paragraphs that appear too frequently
        cleaned_paras = [p for p in refined_paras if paragraph_counts[p] < threshold]

        # Rejoin them into final text
        final_text = "\n".join(cleaned_paras)

        # — Advanced metadata —
        # Language detection
        lang = detect_lang(final_text)

        # Attempt date parse from directory structure (optional)
        iso_date, year_int, month_int = parse_date_from_path(js_path.relative_to(in_dir))

        # Named entities
        named_ents = extract_entities_spacy(final_text, lang)
        # Keywords
        keywords = extract_keywords_yake(final_text, lang)
        # Summary
        summ = simple_summary(final_text, lang)

        # Build final record
        final_record = {
            "doc_id": data.get("doc_id", ""),
            "filename": data.get("filename", ""),
            "domain": "ethz.ch",  # or parse from path if you prefer
            "language": lang,
            "title": data.get("title", ""),
            "date": iso_date,
            "year": year_int,
            "month": month_int,
            "source": "ETH News",
            # final text after disclaimers + repeated paragraphs removed
            "main_content": final_text,
            "paragraphs_original": data.get("paragraphs", []),
            "paragraphs_cleaned": cleaned_paras,
            "named_entities": named_ents,
            "keywords": keywords,
            "summary": summ,
            "text_stats": {
                "char_count": len(final_text),
                "word_count": len(final_text.split()),
                "paragraph_count": len(cleaned_paras)
            },
            "semantic_chunk_hints": [
                {"type": "paragraph_boundaries", "count": len(cleaned_paras)}
            ],
            "embedding_vector": [],
            "doc_embedding": []
        }

        # Output path
        rel_path = js_path.relative_to(in_dir)
        out_path = out_dir / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(final_record, f, indent=2, ensure_ascii=False)
            out_count += 1
            logging.info(f"Processed => {out_path}")
        except Exception as e:
            logging.error(f"Error writing {out_path}: {e}")

    dur = time.time() - start_time
    logging.info(f"Completed advanced cleaning for {out_count} docs in {dur:.2f}s.")

if __name__ == "__main__":
    main()
