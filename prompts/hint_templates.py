"""
Hint templates loaded from data/hint_templates.json.

  HINT_TEMPLATES["<hint_type>"]["<lang>"]  →  format string with {wrong}

  get_conditions("<dataset>")  →  list of (name, hint_type, hint_lang) tuples
                                  for that dataset's experimental conditions
"""
import json
from pathlib import Path

DATA_FILE = Path(__file__).resolve().parents[1] / "data" / "hint_templates.json"
_data = json.loads(DATA_FILE.read_text())

HINT_TEMPLATES = _data["hint_templates"]


def get_conditions(dataset: str):
    """Return list of (cond_name, hint_type, hint_lang) for the given dataset."""
    if dataset not in _data["conditions_per_dataset"]:
        raise KeyError(f"No conditions defined for dataset='{dataset}'. "
                       f"Known: {list(_data['conditions_per_dataset'].keys())}")
    return [(c["name"], c["hint_type"], c["hint_lang"])
            for c in _data["conditions_per_dataset"][dataset]]
