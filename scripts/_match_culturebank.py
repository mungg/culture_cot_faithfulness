"""
Attach a `culturebank_reference` field to each item by retrieving the
top-K most similar CultureBank entries for the same target culture.

This is a one-shot script — re-run if data/*.json items change or if
CultureBank is updated.

Matching:
  - Filter CultureBank rows whose `cultural group` contains the target
    culture keyword (e.g. "korean" matches "Korean Americans", "Korean men", ...)
  - TF-IDF + cosine similarity between (question + options) and a CB entry's
    (context, goal, relation, actor_behavior, recipient_behavior, other, topic).
  - Keep top-K with similarity > MIN_SIM.
"""
import json
from pathlib import Path

from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TOP_K = 2
MIN_SIM = 0.05

CULTURE_KEYWORDS = {
    "Korean": ["korean"],
    "American": ["american"],
    "German": ["german"],
    "Polish": ["polish"],
}


def cb_text(r):
    parts = [r.get("context", ""), r.get("goal", ""), r.get("relation", ""),
             r.get("actor_behavior", ""), r.get("recipient_behavior", ""),
             r.get("other_descriptions", ""), r.get("topic", "")]
    return " ".join(p for p in parts if p and p != "None")


def main():
    ds = load_dataset("SALT-NLP/CultureBank")
    combined = list(ds["tiktok"]) + list(ds["reddit"])
    print(f"Loaded CultureBank: {len(combined)} rows")

    cb_by_culture = {}
    for cu, kws in CULTURE_KEYWORDS.items():
        matched = []
        for i, r in enumerate(combined):
            g = (r["cultural group"] or "").lower()
            if any(k in g for k in kws):
                matched.append({
                    "cb_idx": i,
                    "cultural_group": r["cultural group"],
                    "topic": r["topic"],
                    "context": r["context"],
                    "actor_behavior": r["actor_behavior"],
                    "other_descriptions": r["other_descriptions"],
                    "_text": cb_text(r),
                })
        cb_by_culture[cu] = matched
        print(f"  {cu}: {len(matched)} candidate CB entries")

    def best_matches(item, candidates):
        query = item["question_en"] + " " + " ".join(item["options"])
        texts = [c["_text"] for c in candidates]
        if not texts:
            return []
        vec = TfidfVectorizer(stop_words="english", max_features=3000)
        try:
            m = vec.fit_transform([query] + texts)
        except ValueError:
            return []
        sims = cosine_similarity(m[0:1], m[1:]).ravel()
        top = sims.argsort()[::-1][:TOP_K]
        out = []
        for i in top:
            if sims[i] < MIN_SIM:
                continue
            c = candidates[i]
            out.append({
                "cb_idx": c["cb_idx"],
                "cultural_group": c["cultural_group"],
                "topic": c["topic"],
                "context": c["context"],
                "actor_behavior": c["actor_behavior"],
                "similarity": float(sims[i]),
            })
        return out

    for f in ["korean.json", "american.json", "german.json", "polish.json"]:
        items = json.load(open(DATA / f))
        cu = items[0]["culture"]
        candidates = cb_by_culture[cu]
        for it in items:
            it["culturebank_reference"] = best_matches(it, candidates)
        json.dump(items, open(DATA / f, "w"), ensure_ascii=False, indent=2)
        n = sum(1 for it in items if it["culturebank_reference"])
        print(f"  {f}: {n}/{len(items)} matched")

    all_items = []
    for f in ["korean.json", "american.json", "german.json", "polish.json"]:
        all_items.extend(json.load(open(DATA / f)))
    json.dump(all_items, open(DATA / "cultural_items_all.json", "w"), ensure_ascii=False, indent=2)
    print(f"Total: {len(all_items)} items")


if __name__ == "__main__":
    main()
