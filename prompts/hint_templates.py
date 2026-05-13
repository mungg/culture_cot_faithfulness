"""
Hint templates loaded from data/hint_templates.json so they live alongside
the dataset and can be edited as data, not code.

Each hint points to a WRONG answer (chosen dynamically per (model, item)).
"""
import json
from pathlib import Path

DATA_FILE = Path(__file__).resolve().parents[1] / "data" / "hint_templates.json"
_data = json.loads(DATA_FILE.read_text())

HINT_TEMPLATES = _data["hint_templates"]
CONDITIONS = [(c["name"], c["hint_type"], c["hint_lang"]) for c in _data["conditions"]]
