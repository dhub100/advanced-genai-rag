import pdfplumber
import re
import json
import os
import argparse

def fix_ocr_typos(text):
    """
    Fix only the specific OCR artifacts that differ from the final 'correct version.'
    We remove the broad fallback replacements for 'Z' -> 'ti' and 'T' -> 'll', because
    we actually want many intentional 'Z' and 'T' in the final text (like 'ElioT', 'SebasZno').

    Also add 'EllH' -> 'ETH', 'tiurich' -> 'Zurich', etc., and fix smaller artifacts
    (like 'culng-edge' -> 'cutting-edge', 'swiwly' -> 'swiftly', etc.) to match exactly
    the final JSON text.
    """
    replacements = [
        # Remove or adapt these original ones if they conflict:
        # (r"compeZZve", "competitive"),  # If you no longer see "compeZZve" in your text, you can remove.
        (r"compeZZon", "competition"),
        (r"compeZZve", "competitive"),
        (r"iniZaZve", "initiative"),
        (r"addiZonal", "additional"),
        (r"mulZple", "multiple"),
        (r"opZon", "option"),
        (r"miZgaZng", "mitigating"),
        (r"producZvity", "productivity"),
        (r"projecZons", "projections"),
        (r"direcZon", "direction"),
        (r"capabiliZes", "capabilities"),
        (r"deveZop", "develop"),
        (r"seZngs", "settings"),
        (r"explaZion", "explanation"),
        (r"quesZon", "question"),
        (r"quesZons", "questions"),
        (r"elevaZon", "elevation"),
        (r"elevaZons", "elevations"),
        (r"organizaZon", "organization"),
        (r"acZvely", "actively"),
        (r"experZse", "expertise"),
        (r"collaboraZons", "collaborations"),
        (r"condiZons", "conditions"),
        (r"iniZaZves", "initiatives"),
        (r"insZtuZonal", "institutional"),
        (r"insZtuZon", "institution"),
        (r"insZtute", "institute"),
        (r"parZcular", "particular"),
        (r"addiZon", "addition"),
        (r"conZnues", "continues"),
        (r"acceleraZon", "acceleration"),
        (r"deflecZon", "deflection"),

        # IMPORTANT: Remove these fallback lines so we do NOT replace every 'Z' or every 'T':
        # (r"Z", "ti"),  # <-- Remove this (commented out)
        # (r"T", "ll"),  # <-- Remove this (commented out)

        # We do keep this pipe symbol fix, if needed:
        (r"\|", "l"),

        # Remove "PaTern" -> "Pattern" if not seen, or keep if you do see it:
        (r"PaTern", "Pattern"),

        # Now ADD the explicit fixes to match the final JSON exactly:
        (r"EllH", "ETH"),
        (r"tiurich", "Zurich"),
        (r"lladeus", "Tadeus"),
        (r"llilman", "Tilman"),
        (r"llumor", "Tumor"),
        (r" llanDEM-X", " TanDEM-X"),
        (r"Engimmune llherapeutics", "Engimmune Therapeutics"),

        # Some lines from the extracted text often had small OCR quirks:
        (r"swiwly", "swiftly"),
        (r"culng-edge", "cutting-edge"),
        (r"shiw", "shift"),
        (r"pallern", "pattern"),

        # If your text turned "Anthony Patt" into "Anthony Pall", fix it if needed:
        (r"Anthony Pall", "Anthony Patt"),

        # Fix "Net tiero" -> "Net Zero"
        (r"Net tiero", "Net-Zero"),  # for Q12
        (r"Net tiero", "Net Zero"),  # if you have multiple occurrences, or unify them

        # If your text turned TCRs into llCRs:
        (r"llCRs", "TCRs"),

        # If your text turned "ElioT" into "Elioll", etc.:
        (r"Elioll Ash", "ElioT Ash"),
        (r"Sebastino Cantalupo", "SebasZno Cantalupo"),
        (r"Christian Degen", "ChrisZan Degen"),

        # In case some lines had "Thomas" or "Tobias" incorrectly:
        (r"llobias Donner", "Tobias Donner"),

        # If your text had "Rachel Granoe" spelled differently, correct as needed:
        # (Add or remove lines if you see more tiny mismatches.)
    ]
    for pat, repl in replacements:
        text = re.sub(pat, repl, text)
    return text



# Patch for split Q12 (specific to your file structure)
def merge_q12(qa_pairs):
    for i, qa in enumerate(qa_pairs):
        if qa["id"] == 12 and qa["answer"].startswith("over the years"):
            # Merge question and answer
            qa["question"] = qa["question"].rstrip("?") + " over the years?"
            qa["answer"] = re.sub(r"^over the years\?\s*", "", qa["answer"])
    return qa_pairs



def main():
    parser = argparse.ArgumentParser(
        description="Extract Q&A pairs from a PDF benchmark file"
    )
    parser.add_argument(
        "pdf_path",
        help="Path to the input PDF file (e.g., ../benchmark/BenchmarkQuestionsAnswers.pdf)"
    )
    parser.add_argument(
        "output_path",
        help="Path to the output JSON file (e.g., ../data_cleaned/benchmark_qa.json)"
    )
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # Extract text from PDF
    with pdfplumber.open(args.pdf_path) as pdf:
        all_text = "\n".join(page.extract_text() for page in pdf.pages)

    # Clean OCR issues line by line
    lines = [fix_ocr_typos(line.strip()) for line in all_text.splitlines()]

    qa_pairs = []
    i = 0
    while i < len(lines):
        q_match = re.match(r"^(\d+)\.\s+(.+)", lines[i])
        if q_match:
            q_id = int(q_match.group(1))
            question = q_match.group(2).strip()

            # Read answer/notes (multi-line until next question)
            answer_lines = []
            j = i + 1
            while j < len(lines):
                if re.match(r"^\d+\.\s", lines[j]):  # Next question => break
                    break
                if lines[j] == "":
                    j += 1
                    continue
                answer_lines.append(lines[j])
                j += 1

            # Separate answer vs. notes
            answer, notes = [], []
            is_note = False
            for l in answer_lines:
                # If line looks like scoring/explanation => treat as note
                if any(keyword in l.lower() for keyword in [
                    'score', 'point', 'criterion', 'criteria', 'explanation', 'should',
                    'note', 'this is about', 'deflect', 'ambiguous', 'answer should'
                ]):
                    is_note = True
                if is_note:
                    notes.append(l)
                else:
                    answer.append(l)

            # Sometimes the answer is just long lines, and "notes" might not be actual notes
            if notes and not any(kw in notes[0].lower() for kw in
                                 ['score', 'point', 'criterion', 'criteria', 'deflect', 'ambiguous',
                                  'explanation', 'answer should', 'note']):
                answer += notes
                notes = []

            clean_answer = " ".join(answer).strip()
            # Optionally add final period if it's a multi-word answer missing punctuation
            if clean_answer and not clean_answer.endswith(('.', '?', '!', '"', '"', "'")) \
                    and len(clean_answer.split()) > 3:
                clean_answer += "."

            qa = {
                "id": q_id,
                "question": question,
                "answer": clean_answer
            }
            if notes:
                qa["notes"] = " ".join(notes).strip()

            qa_pairs.append(qa)
            i = j
        else:
            i += 1

    # Patch Q12 if needed
    qa_pairs = merge_q12(qa_pairs)

    # Final polish
    for qa in qa_pairs:
        qa["question"] = qa["question"].strip()
        qa["answer"] = qa["answer"].strip()
        if "notes" in qa:
            qa["notes"] = qa["notes"].strip()

    # Save to JSON
    with open(args.output_path, "w", encoding="utf-8") as fout:
        json.dump(qa_pairs, fout, indent=2, ensure_ascii=False)

    # Print first 3 to verify
    print(json.dumps(qa_pairs[:3], indent=2, ensure_ascii=False))
    print(f"\nSuccessfully extracted {len(qa_pairs)} Q&A pairs to {args.output_path}")


if __name__ == "__main__":
    main()