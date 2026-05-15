"""
Phase 2 (XSAFETY): refusal-loosening hint conditions, on items where the
baseline RESPONSE was a refusal.

Uses scripts/02_hints.py's data layout but with the safety-specific
templates and an open-ended (non-JSON) generation path.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

sys.path.insert(0, str(Path(__file__).parent))
from _models import get_client  # noqa: E402
from _refusal import is_refusal  # noqa: E402

sys.path.insert(0, str(ROOT / "prompts"))
from hint_templates import HINT_TEMPLATES, get_conditions  # noqa: E402


def build_prompt_with_hint(item, hint_type, hint_lang):
    hint_text = HINT_TEMPLATES[hint_type][hint_lang]
    # The cue is appended AFTER the user prompt, framing it as the user's own
    # follow-up commentary on why they think the request should be answered.
    return f"{item['prompt']}\n\n{hint_text}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemini/gemini-2.5-flash")
    ap.add_argument("--baseline", required=True, help="run-name of baseline_safety_<name>.json")
    ap.add_argument("--run-name", default="safety_run1")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--want-thinking", action="store_true")
    ap.add_argument("--conditions", nargs="+", default=None,
                    help="optional subset of condition names")
    ap.add_argument("--limit-items", type=int, default=None,
                    help="cap baseline-refused items per language (smoke)")
    args = ap.parse_args()

    baseline = json.load(open(RESULTS_DIR / f"baseline_safety_{args.baseline}.json"))
    refused_by_id = {b["id"]: b for b in baseline if b["refusal"]}
    items = list(refused_by_id.values())

    by_lang_total = defaultdict(int)
    for b in baseline:
        by_lang_total[b["language"]] += 1
    print("Baseline refusal counts (filter for Phase 2):")
    for lang in sorted({b["language"] for b in baseline}):
        n_ref = sum(1 for b in baseline if b["language"] == lang and b["refusal"])
        print(f"  {lang}: {n_ref}/{by_lang_total[lang]}")

    if args.limit_items:
        bucket = defaultdict(list)
        for it in items:
            bucket[it["language"]].append(it)
        items = sum((b[:args.limit_items] for b in bucket.values()), [])

    conds = get_conditions("xsafety")
    if args.conditions:
        conds = [c for c in conds if c[0] in args.conditions]
    # Drop the baseline pseudo-condition: Phase 2 only runs the hint variants.
    conds = [c for c in conds if c[0] != "baseline"]

    client = get_client(args.model)
    total = len(items) * len(conds)
    print(f"\nHint sweep | model={client.name} | items={len(items)} | conds={len(conds)} | total gens={total}")
    print(f"Conditions: {[c[0] for c in conds]}\n", flush=True)

    out_path = RESULTS_DIR / f"hints_safety_{args.run_name}.json"

    # Resume support: skip (id, condition) pairs already saved.
    existing_keys = set()
    existing_records = []
    if out_path.exists():
        try:
            existing_records = json.load(open(out_path))
            existing_keys = {(r["id"], r["condition"]) for r in existing_records}
            print(f"Resume: {len(existing_keys)} existing (id, condition) records in {out_path.name}",
                  flush=True)
        except (json.JSONDecodeError, KeyError):
            print(f"Warning: {out_path.name} unreadable; overwriting.", flush=True)
            existing_records = []
            existing_keys = set()

    out = list(existing_records)
    k = 0
    for item in items:
        for cond_name, hint_type, hint_lang in conds:
            k += 1
            if (item["id"], cond_name) in existing_keys:
                print(f"  [{k:4d}/{total}] {item['id']} {cond_name}: skip (cached)", flush=True)
                continue
            prompt = build_prompt_with_hint(item, hint_type, hint_lang)
            r = client.generate_text(
                prompt,
                temperature=args.temperature, max_tokens=args.max_tokens,
                want_thinking=args.want_thinking,
            )
            resp = r["text"]
            refused = is_refusal(resp, item["language"])
            rec = {
                "id": item["id"],
                "language": item["language"],
                "category": item["category"],
                "prompt": item["prompt"],
                "hint_text": HINT_TEMPLATES[hint_type][hint_lang],
                "prompt_with_hint": prompt,
                "condition": cond_name,
                "hint_type": hint_type,
                "hint_lang": hint_lang,
                "response": resp,
                "thinking": r["thinking"],
                "refusal": refused,
                "baseline_refused": True,
                "model": client.name,
            }
            out.append(rec)
            # Incremental save so a hang / crash doesn't lose progress.
            out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
            flag = "REFUSE" if refused else "COMPLY"
            print(f"  [{k:4d}/{total}] {item['id']} ({item['language']}) {cond_name}: {flag}", flush=True)

    # Final write (idempotent — file already saved incrementally)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
