"""
Run an end-to-end experiment pipeline for one model/dataset combination.

Examples:
  python scripts/run_pipeline.py \
      --model hf/CohereForAI/aya-expanse-8b \
      --dataset cultureMCQA --culture american \
      --run-name aya8b_us

  python scripts/run_pipeline.py \
      --model hf/Qwen/Qwen3-8B \
      --dataset xsafety \
      --run-name qwen8b_xsafety
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"


def run_step(args: list[str]) -> None:
    print(f"\n==> {' '.join(args)}", flush=True)
    subprocess.run(args, check=True, cwd=ROOT)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", choices=["cultureMCQA", "gsm8k", "xsafety"], required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--culture", choices=["american", "german", "korean", "polish", "all"], default="all")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--want-thinking", action="store_true")
    ap.add_argument("--baseline-limit", type=int, default=None)
    ap.add_argument("--hint-limit-items", type=int, default=None)
    ap.add_argument("--language", choices=["en", "de", "all"], default="all",
                    help="only used for xsafety")
    ap.add_argument("--conditions", nargs="+", default=None)
    ap.add_argument("--biased-gen-model", default=None,
                    help="optional override for 02_hints.py few-shot biased reasoning generation")
    args = ap.parse_args()

    py = sys.executable

    if args.dataset == "xsafety":
        baseline_cmd = [
            py, str(SCRIPTS_DIR / "01_baseline_safety.py"),
            "--model", args.model,
            "--language", args.language,
            "--run-name", args.run_name,
            "--temperature", str(args.temperature),
        ]
        if args.max_tokens is not None:
            baseline_cmd.extend(["--max-tokens", str(args.max_tokens)])
        if args.baseline_limit is not None:
            baseline_cmd.extend(["--limit", str(args.baseline_limit)])
        if args.want_thinking:
            baseline_cmd.append("--want-thinking")
        run_step(baseline_cmd)

        hints_cmd = [
            py, str(SCRIPTS_DIR / "02_hints_safety.py"),
            "--model", args.model,
            "--baseline", args.run_name,
            "--run-name", args.run_name,
            "--temperature", str(args.temperature),
        ]
        if args.max_tokens is not None:
            hints_cmd.extend(["--max-tokens", str(args.max_tokens)])
        if args.hint_limit_items is not None:
            hints_cmd.extend(["--limit-items", str(args.hint_limit_items)])
        if args.conditions:
            hints_cmd.extend(["--conditions", *args.conditions])
        if args.want_thinking:
            hints_cmd.append("--want-thinking")
        run_step(hints_cmd)

        run_step([py, str(SCRIPTS_DIR / "03_analyze_safety.py"), "--run-name", args.run_name])
        return

    baseline_cmd = [
        py, str(SCRIPTS_DIR / "01_baseline.py"),
        "--model", args.model,
        "--dataset", args.dataset,
        "--run-name", args.run_name,
        "--temperature", str(args.temperature),
    ]
    if args.dataset == "cultureMCQA":
        baseline_cmd.extend(["--culture", args.culture])
    if args.max_tokens is not None:
        baseline_cmd.extend(["--max-tokens", str(args.max_tokens)])
    if args.baseline_limit is not None:
        baseline_cmd.extend(["--limit", str(args.baseline_limit)])
    if args.want_thinking:
        baseline_cmd.append("--want-thinking")
    run_step(baseline_cmd)

    hints_cmd = [
        py, str(SCRIPTS_DIR / "02_hints.py"),
        "--model", args.model,
        "--dataset", args.dataset,
        "--baseline", args.run_name,
        "--run-name", args.run_name,
        "--seed", str(args.seed),
        "--temperature", str(args.temperature),
    ]
    if args.max_tokens is not None:
        hints_cmd.extend(["--max-tokens", str(args.max_tokens)])
    if args.biased_gen_model:
        hints_cmd.extend(["--biased-gen-model", args.biased_gen_model])
    if args.hint_limit_items is not None:
        hints_cmd.extend(["--limit-items", str(args.hint_limit_items)])
    if args.conditions:
        hints_cmd.extend(["--conditions", *args.conditions])
    if args.want_thinking:
        hints_cmd.append("--want-thinking")
    run_step(hints_cmd)

    run_step([py, str(SCRIPTS_DIR / "03_analyze.py"), "--run-name", args.run_name])


if __name__ == "__main__":
    main()
