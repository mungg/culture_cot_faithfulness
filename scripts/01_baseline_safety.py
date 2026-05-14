"""
Phase 1 (XSAFETY): open-ended baseline.
Asks the model each prompt as-is, records the response + thinking + refusal flag.

Model-agnostic: pass --model "<provider>/<name>".
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
from _refusal import is_refusal  # noqa: E402


def build_prompt(item):
    # XSAFETY prompts are issued as-is, in the prompt's own language.
    # No JSON wrapping; we want the model's natural response.
    return item["prompt"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemini/gemini-2.5-flash")
    ap.add_argument("--language", choices=["en", "de", "all"], default="all")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--run-name", default="safety_run1")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--want-thinking", action="store_true")
    args = ap.parse_args()

    items = json.load(open(DATA_DIR / "xsafety" / "items.json"))
    if args.language != "all":
        items = [it for it in items if it["language"] == args.language]
    if args.limit:
        # take first N per language so both langs are represented
        bucket = defaultdict(list)
        for it in items:
            bucket[it["language"]].append(it)
        items = sum((b[:args.limit] for b in bucket.values()), [])

    client = get_client(args.model)
    print(f"Safety baseline | model={client.name} | items={len(items)} | langs={sorted({it['language'] for it in items})}", flush=True)

    out = []
    for k, item in enumerate(items, 1):
        r = client.generate_text(
            build_prompt(item),
            temperature=args.temperature, max_tokens=args.max_tokens,
            want_thinking=args.want_thinking,
        )
        resp = r["text"]
        refused = is_refusal(resp, item["language"])
        out.append({
            "id": item["id"],
            "language": item["language"],
            "category": item["category"],
            "prompt": item["prompt"],
            "response": resp,
            "thinking": r["thinking"],
            "refusal": refused,
            "model": client.name,
        })
        flag = "REFUSE" if refused else "COMPLY"
        print(f"  [{k:4d}/{len(items)}] {item['id']} ({item['language']}) {flag}: {resp[:80].replace(chr(10),' ')}", flush=True)

    out_path = RESULTS_DIR / f"baseline_safety_{args.run_name}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))

    by_lang = defaultdict(lambda: {"n": 0, "refused": 0})
    for r in out:
        by_lang[r["language"]]["n"] += 1
        by_lang[r["language"]]["refused"] += int(r["refusal"])
    print("\nBaseline refusal rate:", flush=True)
    for k, d in by_lang.items():
        print(f"  {k:5s}: {d['refused']}/{d['n']} = {d['refused']/d['n']:.2f}", flush=True)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
