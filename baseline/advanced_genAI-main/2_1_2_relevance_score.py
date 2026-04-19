# 2_1_2_relevance_score.py  – batched scorer, one file per chunk
# --------------------------------------------------------------------
# ①  List chunks that still need a score file:
#       python 2_1_2_relevance_score.py --list-missing
# ②  (Re)score only those chunks          (default):
#       python 2_1_2_relevance_score.py            [--no-answer]
# ③  Optional: choose another chunk folder:
#       python 2_1_2_relevance_score.py --chunks path/to/dir
# ④  Optional: specify Q&A path and score directory:
#       python 2_1_2_relevance_score.py --qa-path path/to/qa.json --score-dir path/to/scores
# --------------------------------------------------------------------
import json, pathlib, argparse
from typing import Optional, List, Dict
import dotenv, os
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

# ── OpenAI client ───────────────────────────────────────────────────
dotenv.load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# ── schema for ONE item inside the returned array ──────────────────
class PairScore(BaseModel):
    question_id: int
    relevance_score:  float = Field(..., ge=0, le=1)
    relevance_reason: Optional[str]

# ── system prompt  ─────────────────────────────────────────────────
SYS_PROMPT = """
Return ONE JSON array.  Each array element MUST be:
{
  "question_id": <int>,            // as given below (1‥25)
  "relevance_score": <float 0-1>,
  "relevance_reason": "<≤200 chars>"
}
Do not add other keys. Do not output markdown.
Scoring rubric
 1.0 exact answer · 0.8 answer verbatim · 0.5 partial
 0.2 tangential · 0.0 unrelated
""".strip()

# ── prompt builder (chunk + 25 questions) ──────────────────────────
def build_user_prompt(chunk: dict,
                      qa_items: List[dict],
                      with_answer: bool) -> str:
    lines = [
        "### Chunk ID",
        chunk["metadata"]["chunk_id"],
        "",
        "### Chunk text",
        chunk["text"],
        "",
        "### Questions (25)"
    ]
    for q in qa_items:
        lines.append(f"{q['id']}. EN: {q['question']} / DE: {q['question_de']}")
        if with_answer:
            lines.append(f"   answers → EN: {q['answer']} / DE: {q['answer_de']}")
    return "\n".join(lines).strip()

# ── NEW helper: which chunks *still* need a score file? ────────────
def chunks_missing_scores(chunk_dir: pathlib.Path,
                          score_dir: pathlib.Path) -> List[pathlib.Path]:
    """Return raw-chunk JSONs that have no score file yet."""
    todo: List[pathlib.Path] = []
    for fp in chunk_dir.glob("*.json"):
        with fp.open(encoding="utf-8") as f:
            cid = json.load(f)["metadata"]["chunk_id"]
        if not (score_dir / f"{cid}.json").exists():
            todo.append(fp)
    return sorted(todo)

# ── main ────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    # --- mode flags -------------------------------------------------
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--list-missing", action="store_true",
                      help="Only list chunk-IDs that lack a score file; no API calls.")
    mode.add_argument("--run",          action="store_true",
                      help="Run scoring on missing chunks (default).")
    # --- other flags ------------------------------------------------
    ap.add_argument("--with-answer", dest="with_answer", action="store_true")
    ap.add_argument("--no-answer",  dest="with_answer", action="store_false")
    ap.set_defaults(with_answer=True)
    ap.add_argument("--chunks", default="subsample/semantic_chunk",
                    help="Folder containing raw chunk JSONs")
    ap.add_argument("--qa-path", default="benchmark/benchmark_qa_bilingual_with_semantic_chunks.json",
                    help="Path to the Q&A benchmark JSON file")
    ap.add_argument("--score-dir", default=None,
                    help="Output directory for score files (default: benchmark/score/<chunk_folder_name>)")
    args = ap.parse_args()

    # -------- paths -------------------------------------------------
    qa_path   = pathlib.Path(args.qa_path)
    chunk_dir = pathlib.Path(args.chunks).resolve()

    # ——-- derive score_dir if not explicitly provided --------------
    if args.score_dir:
        score_dir = pathlib.Path(args.score_dir)
    else:
        chunk_folder_name = chunk_dir.name
        stripped_name     = chunk_folder_name.removesuffix("_chunk")
        score_dir         = pathlib.Path("benchmark/score") / stripped_name
    score_dir.mkdir(parents=True, exist_ok=True)

    # ---------- determine what is missing ---------------------------
    todo_files = chunks_missing_scores(chunk_dir, score_dir)

    # ---------- MODE 1: just list & exit ----------------------------
    if args.list_missing:
        if not todo_files:
            print("Everything already scored – nothing missing.")
        else:
            print(f"{len(todo_files)} chunk(s) still need scoring:")
            for fp in todo_files:
                with fp.open(encoding="utf-8") as f:
                    print("-", json.load(f)["metadata"]["chunk_id"])
        return  # ←–––––––––––––––––––––  no token usage

    # ---------- (implicit) MODE 2: run scoring ----------------------
    if not todo_files:
        print("All chunks already have a score file – nothing to do.")
        return

    qa_items_all = json.load(qa_path.open(encoding="utf-8"))
    print(f"{len(todo_files)} chunk(s) will be (re)scored …\n")

    # ---------------- iterate over UN-scored chunks -----------------
    for chunk_fp in todo_files:
        raw   = json.load(chunk_fp.open(encoding="utf-8"))
        cid   = raw["metadata"]["chunk_id"]
        out_path = score_dir / f"{cid}.json"

        # ---- build & send prompt -----------------------------------
        user_prompt = build_user_prompt(raw, qa_items_all, args.with_answer)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYS_PROMPT},
                {"role": "user",   "content": user_prompt}
            ]
        )

        # ---- parse / validate --------------------------------------
        try:
            arr = json.loads(resp.choices[0].message.content)
        except json.JSONDecodeError as e:
            print("✗ JSON decode error for", cid, "→", e)
            continue

        chunk_result: Dict[str, dict] = {}
        for item in arr:
            try:
                ps = PairScore(**item)
            except ValidationError as ve:
                print("✗ validation error for", cid, item, "→", ve)
                break
            chunk_result[str(ps.question_id)] = {
                "relevance_score": ps.relevance_score,
                "relevance_reason": ps.relevance_reason
            }
        else:  # executes only if we did *not* break out of the loop
            if len(chunk_result) != 25:
                print(f"✗ {cid}: expected 25 items, got {len(chunk_result)} – skipping")
                continue
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(chunk_result, f, ensure_ascii=False, indent=2)
            print("✓ scored", cid)

if __name__ == "__main__":
    main()