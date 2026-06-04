import json
from pathlib import Path


def root():
    return Path(__file__).resolve().parents[1]


def load_config():
    path = Path(__file__).with_name("config.json")
    return json.loads(path.read_text(encoding="utf-8"))
