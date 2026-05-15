"""
Submit vLLM-backed reruns for datasets that are slow or unstable under local HF.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BATCH_SCRIPT = ROOT / "slurm" / "run_vllm_pipeline.sbatch"

DEFAULT_TIMES = {
    "cultureMCQA": "01:30:00",
    "gsm8k": "01:15:00",
    "xsafety": "02:00:00",
}


def slugify_model(model: str) -> str:
    return model.lower().replace("/", "_").replace("-", "_").replace(".", "_")


def submit_job(args, *, model: str, dataset: str, culture: str | None = None) -> None:
    run_name = f"hf_{slugify_model(model)}_{dataset}"
    if dataset == "cultureMCQA":
        if not culture:
            raise ValueError("culture is required for cultureMCQA jobs")
        run_name = f"{run_name}_{culture}"
    export_vars = {
        "ALL": None,
        "ROOT_DIR": str(ROOT),
        "DATASET": dataset,
        "CULTURE": culture or "",
        "RUN_NAME": run_name,
        "TEMPERATURE": str(args.temperature),
        "SEED": str(args.seed),
        "MAX_TOKENS": str(args.max_tokens) if args.max_tokens is not None else "",
        "HF_HOME": args.hf_home or "",
        "PIPELINE_PYTHON_BIN": args.pipeline_python_bin,
        "VLLM_PYTHON_BIN": args.vllm_python_bin,
        "VLLM_MODEL_SPEC": model,
        "SERVED_MODEL_NAME": model,
        "LANGUAGE": args.language if dataset == "xsafety" else "",
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
        "--time", args.time or DEFAULT_TIMES[dataset],
        "--cpus-per-task", str(args.cpus_per_task),
        "--mem", args.mem,
        "--gres", f"gpu:{args.gpu_type}:{args.gpus}" if args.gpu_type not in {None, '', 'any'} else f"gpu:{args.gpus}",
        "--export", export_arg,
        str(BATCH_SCRIPT),
    ]
    if args.constraint:
        cmd[1:1] = ["--constraint", args.constraint]
    if args.mail_user:
        cmd.extend(["--mail-user", args.mail_user])
    if args.mail_type:
        cmd.extend(["--mail-type", args.mail_type])
    print("Submitting:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True,
                    help="HF repo names to serve through vLLM, e.g. Qwen/Qwen3-8B")
    ap.add_argument("--datasets", nargs="+", choices=["cultureMCQA", "gsm8k", "xsafety"], required=True)
    ap.add_argument("--cultures", nargs="+", choices=["american", "german", "korean", "polish"],
                    default=["american", "german", "korean", "polish"])
    ap.add_argument("--language", default="all", choices=["en", "de", "all"])
    ap.add_argument("--partition", default="gpu-preempt")
    ap.add_argument("--account", required=True)
    ap.add_argument("--time", default=None)
    ap.add_argument("--mem", default="64G")
    ap.add_argument("--cpus-per-task", type=int, default=8)
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--gpu-type", default="any")
    ap.add_argument("--constraint", default=None)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-dir", default=str(ROOT / "slurm_logs"))
    ap.add_argument("--hf-home", default=None)
    ap.add_argument("--pipeline-python-bin", default="/work/pi_miyyer_umass_edu/jrussell/luffy/bin/python")
    ap.add_argument("--vllm-python-bin", default="/work/pi_miyyer_umass_edu/jrussell/vllm_env/bin/python")
    ap.add_argument("--mail-user", default=None)
    ap.add_argument("--mail-type", default="END,FAIL")
    args = ap.parse_args()

    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    for model in args.models:
        for dataset in args.datasets:
            if dataset == "cultureMCQA":
                for culture in args.cultures:
                    submit_job(args, model=model, dataset=dataset, culture=culture)
            else:
                submit_job(args, model=model, dataset=dataset)


if __name__ == "__main__":
    main()
