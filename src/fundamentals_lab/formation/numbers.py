"""One artifact file, four commands writing their own section of it.

Each command owns a top-level key and leaves the others alone, so they can be run
in any order and re-run individually without losing what the others measured.
"""

import json

from fundamentals_lab.config import FORMATION_NUMBERS_JSON, OUTPUT_DIR


def load() -> dict:
    if not FORMATION_NUMBERS_JSON.exists():
        return {}
    return json.loads(FORMATION_NUMBERS_JSON.read_text())


def save(section: str, payload: dict) -> None:
    numbers = load()
    numbers[section] = payload
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FORMATION_NUMBERS_JSON.write_text(json.dumps(numbers, indent=2) + "\n")
