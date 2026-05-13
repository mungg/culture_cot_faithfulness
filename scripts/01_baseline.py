"""
Phase 1: Baseline — ask the model each question with NO hint.
Records the model's prediction per item.

Model-agnostic: pass --model "<provider>/<name>", e.g.
    --model gemini/gemini-2.5-flash
    --model groq/llama-3.3-70b-versatile
    --model groq/qwen-2.5-32b
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(Path(__file__).parent))
from _models import get_client  # noqa: E402

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemini/gemini-2.5-flash",
                    help="provider/model, e.g. gemini/gemini-2.5-flash, groq/llama-3.3-70b-versatile")
    ap.add_argument("--dataset", choices=["cultureMCQA", "gsm8k"], default="cultureMCQA",
                    help="which MCQA dataset to baseline (XSAFETY uses a different pipeline)")
    ap.add_argument("--culture", choices=["korean", "american", "german", "polish", "all"], default="all")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--run-name", default="run1")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--want-thinking", action="store_true",
                    help="include thinking trace (Gemini 2.5+; ignored otherwise)")
    args = ap.parse_args()

    if args.dataset == "cultureMCQA":
        cdir = DATA_DIR / "cultureMCQA"
        items = json.load(open(cdir / ("items_all.json" if args.culture == "all" else f"{args.culture}.json")))
    else:  # gsm8k
        items = json.load(open(DATA_DIR / "gsm8k" / "items.json"))

    if args.limit:
        bucket = defaultdict(list)
        for it in items:
            key = it.get("culture", it.get("language", "default"))
            bucket[key].append(it)
        items = sum((b[:args.limit] for b in bucket.values()), [])

    client = get_client(args.model)
    print(f"Baseline | model={client.name} | dataset={args.dataset} | items={len(items)}", flush=True)

    out = []
    for k, item in enumerate(items, 1):
        r = client.generate_structured(
            build_prompt(item), RESPONSE_SCHEMA,
            temperature=args.temperature, max_tokens=args.max_tokens,
            want_thinking=args.want_thinking,
        )
        answer = (r["json"] or {}).get("answer")
        reasoning = (r["json"] or {}).get("reasoning", "")
        out.append({
            "id": item["id"],
            "culture": item.get("culture"),
            "correct": item["correct"],
            "options": item["options"],
            "question_en": item["question_en"],
            "predicted_answer": answer,
            "reasoning": reasoning,
            "thinking": r["thinking"],
            "model": client.name,
        })
        print(f"  [{k:4d}/{len(items)}] {item['id']}: pred={answer} gold={item['correct']}", flush=True)

    out_path = RESULTS_DIR / f"baseline_{args.run_name}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))

    by_key = defaultdict(lambda: {"n": 0, "correct": 0})
    for r in out:
        k = r.get("culture") or "all"
        by_key[k]["n"] += 1
        if r["predicted_answer"] == r["correct"]:
            by_key[k]["correct"] += 1
    print("\nBaseline accuracy:", flush=True)
    for k, d in by_key.items():
        print(f"  {k:10s}: {d['correct']}/{d['n']} = {d['correct']/d['n']:.2f}", flush=True)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
