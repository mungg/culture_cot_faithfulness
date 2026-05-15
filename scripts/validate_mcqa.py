"""
Lightweight validator for CultureMCQA files and translated hint templates.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def validate_dataset(path: Path) -> list[str]:
    issues: list[str] = []
    items = json.load(open(path))
    for item in items:
        if len(item.get("options", [])) != 4:
            issues.append(f"{path.name}:{item['id']} expected 4 options, found {len(item.get('options', []))}")
        labels = []
        for opt in item.get("options", []):
            m = re.match(r"^\(([A-D])\)\s+", opt)
            labels.append(m.group(1) if m else None)
        if labels != ["A", "B", "C", "D"]:
            issues.append(f"{path.name}:{item['id']} option labels malformed: {labels}")
        correct = item.get("correct", "")
        if correct not in {"A", "B", "C", "D"}:
            issues.append(f"{path.name}:{item['id']} invalid correct label: {correct!r}")
        if "culturebank_reference" not in item or not item["culturebank_reference"]:
            issues.append(f"{path.name}:{item['id']} missing CultureBank reference")
    return issues


def validate_hint_templates(path: Path) -> list[str]:
    payload = json.load(open(path))
    issues: list[str] = []
    templates = payload["hint_templates"]
    for hint_type in ["cultural_authority", "general_authority", "social", "indirect"]:
        for lang in ["en", "ko", "de", "pl"]:
            if lang not in templates.get(hint_type, {}):
                issues.append(f"hint_templates.json missing {hint_type}.{lang}")
    return issues


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cultureMCQA", "gsm8k", "xsafety", "all"], default="all")
    ap.add_argument("--culture", choices=["american", "german", "korean", "polish", "all"], default="all")
    args = ap.parse_args()

    all_issues: list[str] = []
    if args.dataset in {"cultureMCQA", "all"}:
        if args.culture == "all":
            names = ["american.json", "german.json", "korean.json", "polish.json", "items_all.json"]
        else:
            names = [f"{args.culture}.json"]
        for name in names:
            all_issues.extend(validate_dataset(DATA_DIR / "cultureMCQA" / name))
        all_issues.extend(validate_hint_templates(DATA_DIR / "hint_templates.json"))

    if all_issues:
        print("Validation issues:")
        for issue in all_issues:
            print(f"  - {issue}")
    else:
        print("No validation issues found.")


if __name__ == "__main__":
    main()
