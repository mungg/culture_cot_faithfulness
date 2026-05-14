"""
Sanity-check the environment before running real experiments.

Usage:  python scripts/setup_check.py [--model provider/name]

Verifies for the chosen provider:
  1. Required Python package importable.
  2. Required environment variable / auth present.
  3. A real round-trip call works for both `generate_structured` and
     `generate_text`.

Exits 0 on success.  Run before handing off the repo or before running
a long sweep.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _models import get_client  # noqa: E402

PROVIDER_REQS = {
    "gemini": {
        "package": "google.genai",
        "env": ["VERTEX_PROJECT (optional, default = author's project)",
                "VERTEX_LOCATION (optional, default = us-central1)",
                "VERTEX_USER_LABEL (optional, billing label)"],
        "auth_note": "Run `gcloud auth application-default login` once.",
    },
    "groq": {
        "package": "groq",
        "env": ["GROQ_API_KEY (required)"],
        "auth_note": "Get a key at https://console.groq.com.",
    },
    "openai": {
        "package": "openai",
        "env": ["OPENAI_API_KEY (required)"],
        "auth_note": "Get a key at https://platform.openai.com.",
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="groq/qwen-2.5-32b",
                    help="provider/model to test (default: groq/qwen-2.5-32b)")
    args = ap.parse_args()

    provider = args.model.split("/", 1)[0]
    print(f"== Setup check for: {args.model} (provider={provider}) ==\n")

    reqs = PROVIDER_REQS.get(provider)
    if not reqs:
        print(f"  unknown provider '{provider}'. Known: {list(PROVIDER_REQS)}")
        sys.exit(2)

    print(f"[1/4] Package importable ({reqs['package']})... ", end="")
    try:
        __import__(reqs["package"])
        print("OK")
    except ImportError as e:
        print(f"FAIL\n  {e}\n  → pip install -r requirements.txt")
        sys.exit(1)

    print(f"[2/4] Auth / env vars:")
    for note in reqs["env"]:
        key = note.split()[0]
        val = os.environ.get(key, "")
        marker = "set" if val else "unset"
        print(f"        {key:30s} {marker}")
    print(f"        ({reqs['auth_note']})")

    print(f"[3/4] generate_structured round-trip... ", end="", flush=True)
    client = get_client(args.model)
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string", "enum": ["A", "B"]}},
        "required": ["answer"],
    }
    r = client.generate_structured(
        "Output a JSON object with key 'answer' whose value is the string 'A'.",
        schema, max_tokens=256,
    )
    ans = (r.get("json") or {}).get("answer")
    if ans in {"A", "B"}:
        print(f"OK (got '{ans}')")
    else:
        print(f"FAIL\n  text={r.get('text','')[:120]!r}")
        sys.exit(1)

    print(f"[4/4] generate_text round-trip...       ", end="", flush=True)
    r = client.generate_text("Say 'hi' briefly.", max_tokens=32)
    text = (r.get("text") or "").strip()
    if text:
        print(f"OK ({text[:40]!r})")
    else:
        print("FAIL (empty text)")
        sys.exit(1)

    print(f"\nAll checks passed for {args.model}.")


if __name__ == "__main__":
    main()
