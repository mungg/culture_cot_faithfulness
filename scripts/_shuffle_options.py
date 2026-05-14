"""
One-shot utility: shuffle the option order of every item in a
cultureMCQA file so that the correct letter is uniform across A/B/C/D.

The original files had a strong position bias (48-50 / 50 items had
correct=B), so any baseline accuracy or "robustness" measurement was
confounded with a model's positional preference. After shuffling
(deterministic seed=42), the correct letter is sampled fresh per item
from a uniform distribution.

Usage:
    python scripts/_shuffle_options.py                # all 4 cultures + items_all
    python scripts/_shuffle_options.py --culture korean    # one only
"""
import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CDIR = ROOT / "data" / "cultureMCQA"

OPT_RE = re.compile(r"^\([A-D]\)\s*")


def shuffle_item(item: dict, rng: random.Random) -> dict:
    """Shuffle the 4 options uniformly. Returns a NEW item (does not mutate)."""
    perm = list(range(4))
    rng.shuffle(perm)
    raw = [OPT_RE.sub("", o) for o in item["options"]]
    new_raw = [raw[i] for i in perm]
    new_options = [f"({L}) {t}" for L, t in zip("ABCD", new_raw)]
    old_ci = "ABCD".index(item["correct"])
    new_ci = perm.index(old_ci)
    out = dict(item)
    out["options"] = new_options
    out["correct"] = "ABCD"[new_ci]
    out["_options_shuffled"] = True
    out["_orig_correct"] = item["correct"]  # provenance for audit
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--culture", choices=["korean", "american", "german", "polish", "all"], default="all")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cultures = ["korean", "american", "german", "polish"] if args.culture == "all" else [args.culture]
    rng = random.Random(args.seed)

    all_items = []
    for cu in cultures:
        path = CDIR / f"{cu}.json"
        items = json.load(open(path))
        before = Counter(it["correct"] for it in items)

        # Backup once.
        backup = CDIR / f"{cu}_v1_biased.json"
        if not backup.exists():
            backup.write_text(json.dumps(items, ensure_ascii=False, indent=2))
            print(f"  backup created: {backup.name}")

        shuffled = [shuffle_item(it, rng) for it in items]
        after = Counter(it["correct"] for it in shuffled)

        path.write_text(json.dumps(shuffled, ensure_ascii=False, indent=2))
        print(f"{cu:10s} before={dict(before)} after={dict(after)}")
        all_items.extend(shuffled)

    # Rebuild items_all.json if we shuffled all
    if args.culture == "all":
        all_path = CDIR / "items_all.json"
        if all_path.exists():
            backup = CDIR / "items_all_v1_biased.json"
            if not backup.exists():
                backup.write_text(json.dumps(json.load(open(all_path)), ensure_ascii=False, indent=2))
                print(f"  backup created: {backup.name}")
        all_path.write_text(json.dumps(all_items, ensure_ascii=False, indent=2))
        print(f"\nWrote items_all.json (N={len(all_items)})")
        print("Combined correct distribution:", dict(Counter(it["correct"] for it in all_items)))


if __name__ == "__main__":
    main()
