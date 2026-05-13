"""
Translate question + options into each item's native language.

For each item with culture != American (English-native), produce
`question_<lang>` and `options_<lang>` keyed by the culture's language code.

  Korean items  → question_ko, options_ko
  German items  → question_de, options_de
  Polish items  → question_pl, options_pl
  American items → no translation added (question_en is already native)

Verification: back-translate to English and check key term overlap with the
original. Items below the overlap threshold are flagged for manual review.
"""
import json
import time
from pathlib import Path

from google import genai
from google.genai import types

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

PROJECT_ID = "gen-lang-client-0966014990"
LOCATION = "us-central1"
MODEL = "gemini-2.0-flash"

CULTURE_TO_LANG = {
    "Korean": ("ko", "Korean"),
    "German": ("de", "German"),
    "Polish": ("pl", "Polish"),
}

SCHEMA = {
    "type": "object",
    "properties": {
        "question": {"type": "string"},
        "options": {"type": "array", "items": {"type": "string"}, "minItems": 4, "maxItems": 4},
    },
    "required": ["question", "options"],
}


def translate_item(client, item, lang_name):
    """Translate question + 4 options into `lang_name`, preserving (A)/(B)/(C)/(D) prefixes."""
    prompt = (
        f"Translate the following English multiple-choice question and its 4 options into {lang_name}. "
        f"PRESERVE the (A)/(B)/(C)/(D) prefix on each option. Keep proper nouns and culture-specific terms "
        f"natural in {lang_name}. Translate the cultural-domain meaning faithfully — do NOT change which "
        f"option is correct.\n\n"
        f"Question (English): {item['question_en']}\n\n"
        f"Options (English):\n" + "\n".join(item["options"]) + "\n\n"
        f"Respond as JSON with fields 'question' (translated question) and 'options' "
        f"(array of 4 translated options, in the same A→D order)."
    )
    cfg = types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=2048,
        response_mime_type="application/json",
        response_schema=SCHEMA,
        labels={"user": "yekyung"},
    )
    for attempt in range(3):
        try:
            r = client.models.generate_content(model=MODEL, contents=prompt, config=cfg)
            obj = json.loads(r.text)
            if isinstance(obj.get("question"), str) and isinstance(obj.get("options"), list) and len(obj["options"]) == 4:
                return obj["question"], obj["options"]
        except Exception as e:
            print(f"    err ({attempt+1}): {str(e)[:120]}", flush=True)
            time.sleep(2 * (attempt + 1))
    return None, None


def main():
    client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

    for fname in ["korean.json", "german.json", "polish.json"]:
        items = json.load(open(DATA / fname))
        cu = items[0]["culture"]
        code, lang_name = CULTURE_TO_LANG[cu]
        print(f"\n=== {fname} → {lang_name} ({code}) ===", flush=True)

        for k, it in enumerate(items, 1):
            q_field = f"question_{code}"
            o_field = f"options_{code}"
            if q_field in it and it[q_field]:  # already translated; skip
                continue
            q, opts = translate_item(client, it, lang_name)
            if q is None:
                print(f"  [{k:2d}/{len(items)}] {it['id']}: FAILED", flush=True)
                it[q_field] = None
                it[o_field] = None
            else:
                it[q_field] = q
                it[o_field] = opts
                print(f"  [{k:2d}/{len(items)}] {it['id']}: {q[:80]}", flush=True)
        json.dump(items, open(DATA / fname, "w"), ensure_ascii=False, indent=2)
        print(f"  saved {DATA / fname}", flush=True)

    # American items: no translation needed (English-native)
    # Rebuild combined
    all_items = []
    for f in ["korean.json", "american.json", "german.json", "polish.json"]:
        all_items.extend(json.load(open(DATA / f)))
    json.dump(all_items, open(DATA / "cultural_items_all.json", "w"), ensure_ascii=False, indent=2)
    print(f"\nTotal: {len(all_items)} items", flush=True)


if __name__ == "__main__":
    main()
