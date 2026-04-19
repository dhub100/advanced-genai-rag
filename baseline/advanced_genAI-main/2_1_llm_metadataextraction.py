# 2_1_llm_metadataextraction.py
# --------------------------------------------------------------------
# Usage:  python 2_1_llm_metadataextraction.py \
#           --chunks subsample/semantic_chunk \
#           --meta-dir path/to/output
# --------------------------------------------------------------------
import json, pathlib, argparse, os
from typing import List, Optional, Union, Any
import dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

# ── OpenAI client ───────────────────────────────────────────────────
dotenv.load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# ── Pydantic schema (metadata only) ─────────────────────────────────
class Role(BaseModel):
    person: str
    role: str
    from_: Optional[int] = Field(None, alias="from")
    to:    Optional[int]

class Event(BaseModel):
    label: str
    year: int
    month: Optional[int]
    day:   Optional[int]

class NumericFact(BaseModel):
    name: str
    value: float
    unit: str
    year: Optional[int]

class Entities(BaseModel):
    person:   List[str] = []
    org:      List[str] = []
    location: List[str] = []

class MetadataSchema(BaseModel):
    id: str
    chunk_summary: str
    entities: Entities
    topic_tags:       List[str] = []
    event_dates:      List[Event] = []
    role_annotations: List[Role]  = []
    numeric_facts:    List[NumericFact] = []
    department:       List[str] = []
    document_type:    str
    content_year:     int
    content_month:    int
    initiative:       List[str] = []
    grant_type:       List[str] = []
    model_config = {"populate_by_name": True}

# ── Prompt ──────────────────────────────────────────────────────────
SYSTEM_MSG = """
Return ONE JSON object that validates against the schema.
• Do NOT add or delete keys; use "" or [] if unknown.
• chunk_summary ≤ 30 words, no quotes.
• role_annotations.role must be a normalised title (president, rector, …).
• event_dates numbers only {"year":2024,"month":5,"day":17}, month/day null if unk.
• content_month integer 1-12.

Output must be pure JSON – no markdown – and must not invent facts.
""".strip()

# ── Helper functions ────────────────────────────────────────────────
def load_json_utf8(path: pathlib.Path):
    for enc in ("utf-8", "utf-8-sig"):
        try:
            with path.open(encoding=enc) as f:
                return json.load(f)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Cannot read {path} as UTF-8")

def _dedup(seq: list[str]) -> list[str]:
    seen, out = set(), []
    for s in seq:
        if s not in seen:
            seen.add(s); out.append(s)
    return out

# ---------- universal *deep* lower-casing utility --------------------------- #
def _deep_lower(obj: Any) -> Any:
    """
    Recursively walk any JSON-serialisable structure and
    lower-case *all* strings, deduplicating lists of strings.
    """
    if isinstance(obj, str):
        return obj.lower()

    if isinstance(obj, list):
        lowered = [_deep_lower(v) for v in obj]
        # If this is *now* a list of strings → de-dup
        if all(isinstance(v, str) for v in lowered):
            return _dedup(lowered)
        return lowered

    if isinstance(obj, dict):
        return {k: _deep_lower(v) for k, v in obj.items()}

    return obj  # numbers / None / bool stay unchanged

# --------------------------------------------------------------------------- #
def normalise(meta: MetadataSchema) -> MetadataSchema:
    """
    Fully lower-case every string field (except the primary `id` which
    is already lower-case by construction) and collapse double spaces.
    """
    data = _deep_lower(meta.model_dump())               #  ‹CHANGED›

    # collapse whitespace in the *now* lower-cased summary
    data["chunk_summary"] = " ".join(data["chunk_summary"].split())

    return MetadataSchema(**data)

# ── main ────────────────────────────────────────────────────────────
def main() -> None:
    argp = argparse.ArgumentParser(description="Extract metadata for each chunk")
    argp.add_argument("--chunks", default="subsample/semantic_chunk",
                      help="Folder with raw chunk JSON files")
    argp.add_argument("--meta-dir", default=None,
                      help="Output directory for metadata files (default: benchmark/metadata/<chunk_folder_name>)")
    args = argp.parse_args()

    chunk_dir = pathlib.Path(args.chunks).resolve()

    # ——-- derive meta_dir if not explicitly provided --------------
    if args.meta_dir:
        meta_dir = pathlib.Path(args.meta_dir)
    else:
        # derive a clean sub-folder name (strip trailing '_chunk' if present)
        subfolder = chunk_dir.name.removesuffix("_chunk")
        meta_dir  = pathlib.Path("benchmark/metadata") / subfolder
    meta_dir.mkdir(parents=True, exist_ok=True)

    for fp in sorted(chunk_dir.glob("*.json")):
        raw = load_json_utf8(fp)
        cid = raw["metadata"]["chunk_id"]
        out_path = meta_dir / f"{cid}.json"
        if out_path.exists():                          # skip already processed
            continue

        user_prompt = f"### Chunk ID\n{raw['id']}\n\n### Chunk text\n{raw['text']}"
        response = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": SYSTEM_MSG},
                      {"role": "user",   "content": user_prompt}],
            response_format=MetadataSchema
        )
        meta = normalise(response.choices[0].message.parsed)

        with out_path.open("w", encoding="utf-8") as f:
            json.dump(meta.model_dump(), f, ensure_ascii=False, indent=2)
        print("✓", cid)

if __name__ == "__main__":
    main()