"""
Submit a batch of Slurm jobs for one or more models across datasets.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BATCH_SCRIPT = ROOT / "slurm" / "run_pipeline.sbatch"

DEFAULT_TIMES = {
    "cultureMCQA": "01:30:00",
    "gsm8k": "01:15:00",
    "xsafety": "02:00:00",
}


def slugify_model(model: str) -> str:
    return model.lower().replace("/", "_").replace("-", "_").replace(".", "_")


def walltime_for(dataset: str, override: str | None) -> str:
    return override or DEFAULT_TIMES[dataset]


def submit_job(args, *, model: str, dataset: str, culture: str | None = None, language: str | None = None) -> None:
    run_name = f"{slugify_model(model)}_{dataset}"
    if culture:
        run_name += f"_{culture}"
    if language and language != "all":
        run_name += f"_{language}"

    export_vars = {
        "ALL": None,
        "ROOT_DIR": str(ROOT),
        "MODEL_SPEC": model,
        "DATASET": dataset,
        "RUN_NAME": run_name,
        "TEMPERATURE": str(args.temperature),
        "SEED": str(args.seed),
        "MAX_TOKENS": str(args.max_tokens) if args.max_tokens is not None else "",
        "HF_HOME": args.hf_home or "",
        "PYTHON_BIN": args.python_bin or "",
        "VENV_ACTIVATE": args.venv_activate or "",
        "WANT_THINKING": "1" if args.want_thinking else "0",
        "BASELINE_LIMIT": str(args.baseline_limit) if args.baseline_limit is not None else "",
        "HINT_LIMIT_ITEMS": str(args.hint_limit_items) if args.hint_limit_items is not None else "",
        "BIASED_GEN_MODEL": args.biased_gen_model or "",
        "CULTURE": culture or "",
        "LANGUAGE": language or "",
    }
    export_arg = ",".join(
        key if value is None else f"{key}={value}"
        for key, value in export_vars.items()
    )

    cmd = [
        "sbatch",
        "--job-name", run_name[:128],
        "--output", str(Path(args.log_dir) / f"{run_name}.%j.out"),
        "--partition", args.partition,
        "--account", args.account,
        "--time", walltime_for(dataset, args.time),
        "--cpus-per-task", str(args.cpus_per_task),
        "--mem", args.mem,
        "--gres", f"gpu:{args.gpu_type}:{args.gpus}" if args.gpu_type not in {None, '', 'any'} else f"gpu:{args.gpus}",
        "--export", export_arg,
        str(BATCH_SCRIPT),
    ]
    if args.mail_user:
        cmd.extend(["--mail-user", args.mail_user])
    if args.mail_type:
        cmd.extend(["--mail-type", args.mail_type])
    print("Submitting:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True,
                    help="one or more model specs, e.g. hf/CohereForAI/aya-expanse-8b")
    ap.add_argument("--datasets", nargs="+", default=["cultureMCQA", "gsm8k", "xsafety"])
    ap.add_argument("--cultures", nargs="+", default=["american", "german", "korean", "polish"])
    ap.add_argument("--language", default="all", choices=["en", "de", "all"],
                    help="xsafety language filter")
    ap.add_argument("--partition", default="gpu-preempt")
    ap.add_argument("--account", required=True)
    ap.add_argument("--time", default=None,
                    help="optional walltime override; defaults: cultureMCQA=01:30:00, gsm8k=01:15:00, xsafety=02:00:00")
    ap.add_argument("--mem", default="64G")
    ap.add_argument("--cpus-per-task", type=int, default=8)
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--gpu-type", default="l40s",
                    help="GPU type constraint on Slurm, e.g. l40s/a100/h100. Use 'any' to allow any GPU on the partition.")
    ap.add_argument("--mail-user", default=None)
    ap.add_argument("--mail-type", default="END,FAIL")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-dir", default=str(ROOT / "slurm_logs"))
    ap.add_argument("--hf-home", default=None)
    ap.add_argument("--python-bin", default=None)
    ap.add_argument("--venv-activate", default=None)
    ap.add_argument("--want-thinking", action="store_true")
    ap.add_argument("--baseline-limit", type=int, default=None)
    ap.add_argument("--hint-limit-items", type=int, default=None)
    ap.add_argument("--biased-gen-model", default=None)
    args = ap.parse_args()

    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    for model in args.models:
        for dataset in args.datasets:
            if dataset == "cultureMCQA":
                for culture in args.cultures:
                    submit_job(args, model=model, dataset=dataset, culture=culture)
            elif dataset == "xsafety":
                submit_job(args, model=model, dataset=dataset, language=args.language)
            else:
                submit_job(args, model=model, dataset=dataset)


if __name__ == "__main__":
    main()
