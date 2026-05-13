"""
Run the experiment: for each (item, condition), query the model and save
the structured response + thinking trace.

The model returns JSON: {"reasoning": "...", "answer": "A|B|C|D"}
— guaranteed by Gemini's response_schema, so no regex parsing needed.

Output: results/raw_<run_name>.json
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from google import genai
from google.genai import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "prompts"))
from hint_templates import HINT_TEMPLATES, CONDITIONS  # noqa: E402

# --- Vertex AI config ---
PROJECT_ID = "gen-lang-client-0966014990"
LOCATION = "us-central1"
MODEL_ID = "gemini-2.5-flash"
THINKING_BUDGET = 8000

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Structured output schema — guarantees valid JSON
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string", "description": "Step-by-step reasoning"},
        "answer": {"type": "string", "enum": ["A", "B", "C", "D"]},
    },
    "required": ["reasoning", "answer"],
}


def build_prompt(item, hint_type=None, hint_lang=None):
    q = item["question_en"]
    opts = "\n".join(item["options"])
    prompt = f"{q}\n\n{opts}\n\n"
    if hint_type and hint_lang:
        template = HINT_TEMPLATES[hint_type][hint_lang]
        prompt += template.format(wrong=item["wrong_hint"]) + "\n\n"
    prompt += (
        "Respond with a JSON object containing 'reasoning' "
        "(your step-by-step thinking) and 'answer' (one of A, B, C, D)."
    )
    return prompt


def call_gemini(client, prompt):
    cfg = types.GenerateContentConfig(
        temperature=1.0,
        max_output_tokens=4096,
        response_mime_type="application/json",
        response_schema=RESPONSE_SCHEMA,
        thinking_config=types.ThinkingConfig(
            thinking_budget=THINKING_BUDGET,
            include_thoughts=True,
        ),
        labels={"user": "yekyung"},
    )
    for attempt in range(3):
        try:
            r = client.models.generate_content(model=MODEL_ID, contents=prompt, config=cfg)
            thinking, response_text = "", ""
            for part in r.candidates[0].content.parts:
                if part.thought:
                    thinking += part.text
                else:
                    response_text += part.text
            try:
                obj = json.loads(response_text)
                return obj.get("reasoning", ""), obj.get("answer"), thinking, response_text
            except json.JSONDecodeError:
                return "", None, thinking, response_text
        except Exception as e:
            print(f"  err (attempt {attempt+1}): {str(e)[:120]}", flush=True)
            time.sleep(2 * (attempt + 1))
    return "", None, "", ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--culture", choices=["korean", "american", "german", "polish", "all"], default="all")
    ap.add_argument("--limit", type=int, default=None, help="limit items per culture (for testing)")
    ap.add_argument("--run-name", default="run1")
    args = ap.parse_args()

    if args.culture == "all":
        items = json.load(open(DATA_DIR / "cultural_items_all.json"))
    else:
        items = json.load(open(DATA_DIR / f"{args.culture}.json"))
    if args.limit:
        bucket = defaultdict(list)
        for it in items:
            bucket[it["culture"]].append(it)
        items = sum((b[:args.limit] for b in bucket.values()), [])

    print(f"Loaded {len(items)} items × {len(CONDITIONS)} conditions = {len(items)*len(CONDITIONS)} calls")

    client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
    out_path = RESULTS_DIR / f"raw_{args.run_name}.json"
    results = []
    total = len(items) * len(CONDITIONS)
    k = 0
    for item in items:
        for cond_name, hint_type, hint_lang in CONDITIONS:
            k += 1
            prompt = build_prompt(item, hint_type, hint_lang)
            reasoning, answer, thinking, raw = call_gemini(client, prompt)
            results.append({
                "id": item["id"],
                "culture": item["culture"],
                "correct": item["correct"],
                "wrong_hint": item["wrong_hint"],
                "condition": cond_name,
                "hint_type": hint_type,
                "hint_lang": hint_lang,
                "prompt": prompt,
                "reasoning": reasoning,
                "predicted_answer": answer,
                "thinking": thinking,
                "raw_response": raw,
            })
            print(f"  [{k:4d}/{total}] {item['id']} {cond_name}: answer={answer}", flush=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
