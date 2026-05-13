"""
Phase 2: Hint conditions — run only on items where baseline was CORRECT,
and pick `wrong_hint` dynamically per item:

  wrong_hint = random letter in {A,B,C,D} that is != correct  AND  != baseline_prediction

(If baseline_prediction == correct, this just excludes the gold letter.)

This ensures the hint truly contradicts the model's belief — a proper
sycophancy test (à la Turpin et al. 2024).

Output: results/hints_<run_name>.json
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

from google import genai
from google.genai import types

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

sys.path.insert(0, str(ROOT / "prompts"))
from hint_templates import HINT_TEMPLATES, CONDITIONS  # noqa: E402

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


def pick_wrong_hint(correct, baseline_pred, rng):
    """Pick a letter that's neither gold nor the model's baseline prediction."""
    candidates = [c for c in "ABCD" if c != correct and c != baseline_pred]
    if not candidates:  # only possible if baseline_pred == correct AND we ran out (impossible with 4 options)
        candidates = [c for c in "ABCD" if c != correct]
    return rng.choice(candidates)


def build_prompt(item, hint_type, hint_lang, wrong_letter):
    q = item["question_en"]
    opts = "\n".join(item["options"])
    template = HINT_TEMPLATES[hint_type][hint_lang]
    hint_text = template.format(wrong=wrong_letter)
    return (
        f"{q}\n\n{opts}\n\n{hint_text}\n\n"
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
    ap.add_argument("--baseline", required=True, help="baseline run-name to load")
    ap.add_argument("--run-name", default="run1")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    baseline = json.load(open(RESULTS_DIR / f"baseline_{args.baseline}.json"))
    items_full = {it["id"]: it for it in json.load(open(DATA_DIR / "cultural_items_all.json"))}

    correct_items = [b for b in baseline if b["predicted_answer"] == b["correct"]]
    print(f"Baseline-correct items: {len(correct_items)}/{len(baseline)}", flush=True)

    rng = random.Random(args.seed)
    hint_conds = [(n, ht, hl) for (n, ht, hl) in CONDITIONS if ht is not None]
    print(f"Hint conditions: {len(hint_conds)}  → total calls: {len(correct_items)*len(hint_conds)}", flush=True)

    client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
    out = []
    total = len(correct_items) * len(hint_conds)
    k = 0
    for b in correct_items:
        item = items_full[b["id"]]
        wrong_hint = pick_wrong_hint(b["correct"], b["predicted_answer"], rng)
        for cond_name, hint_type, hint_lang in hint_conds:
            k += 1
            prompt = build_prompt(item, hint_type, hint_lang, wrong_hint)
            reasoning, answer, thinking = call_gemini(client, prompt)
            out.append({
                "id": item["id"], "culture": item["culture"], "correct": item["correct"],
                "baseline_prediction": b["predicted_answer"],
                "wrong_hint": wrong_hint,
                "condition": cond_name, "hint_type": hint_type, "hint_lang": hint_lang,
                "predicted_answer": answer, "reasoning": reasoning, "thinking": thinking,
            })
            print(f"  [{k:4d}/{total}] {item['id']} {cond_name} (wrong_hint={wrong_hint}): pred={answer}", flush=True)

    out_path = RESULTS_DIR / f"hints_{args.run_name}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
