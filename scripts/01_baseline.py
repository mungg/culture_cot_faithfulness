"""
Phase 1: Baseline — ask the model each question with NO hint.
Records the model's prediction per item. Used downstream to (a) filter
items where the model already gets the gold answer, and (b) choose a
sensible `wrong_hint` target for the hint conditions (different from both
the gold answer and the model's baseline prediction).

Output: results/baseline_<run_name>.json
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from google import genai
from google.genai import types

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

PROJECT_ID = "gen-lang-client-0966014990"
LOCATION = "us-central1"
MODEL_ID = "gemini-2.5-flash"
THINKING_BUDGET = 8000

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "answer": {"type": "string", "enum": ["A", "B", "C", "D"]},
    },
    "required": ["reasoning", "answer"],
}


def build_prompt(item):
    q = item["question_en"]
    opts = "\n".join(item["options"])
    return (
        f"{q}\n\n{opts}\n\n"
        "Respond with a JSON object containing 'reasoning' "
        "(your step-by-step thinking) and 'answer' (one of A, B, C, D)."
    )


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
            thinking, resp = "", ""
            for part in r.candidates[0].content.parts:
                if part.thought:
                    thinking += part.text
                else:
                    resp += part.text
            try:
                obj = json.loads(resp)
                return obj.get("reasoning", ""), obj.get("answer"), thinking
            except json.JSONDecodeError:
                return "", None, thinking
        except Exception as e:
            print(f"  err ({attempt+1}): {str(e)[:120]}", flush=True)
            time.sleep(2 * (attempt + 1))
    return "", None, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--culture", choices=["korean", "american", "german", "polish", "all"], default="all")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--run-name", default="run1")
    args = ap.parse_args()

    items = json.load(open(DATA_DIR / ("cultural_items_all.json" if args.culture == "all" else f"{args.culture}.json")))
    if args.limit:
        bucket = defaultdict(list)
        for it in items:
            bucket[it["culture"]].append(it)
        items = sum((b[:args.limit] for b in bucket.values()), [])

    print(f"Baseline run: {len(items)} items, model={MODEL_ID}", flush=True)
    client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

    out = []
    for k, item in enumerate(items, 1):
        reasoning, answer, thinking = call_gemini(client, build_prompt(item))
        out.append({
            "id": item["id"], "culture": item["culture"], "correct": item["correct"],
            "options": item["options"], "question_en": item["question_en"],
            "predicted_answer": answer, "reasoning": reasoning, "thinking": thinking,
        })
        print(f"  [{k:4d}/{len(items)}] {item['id']}: pred={answer} gold={item['correct']}", flush=True)

    out_path = RESULTS_DIR / f"baseline_{args.run_name}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))

    by_culture = defaultdict(lambda: {"n": 0, "correct": 0})
    for r in out:
        by_culture[r["culture"]]["n"] += 1
        if r["predicted_answer"] == r["correct"]:
            by_culture[r["culture"]]["correct"] += 1
    print("\nBaseline accuracy:", flush=True)
    for cu, d in by_culture.items():
        print(f"  {cu:10s}: {d['correct']}/{d['n']} = {d['correct']/d['n']:.2f}", flush=True)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
