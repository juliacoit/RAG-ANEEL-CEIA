import json
from datetime import datetime
from pathlib import Path

LOG_PATH = Path("data/logs/logs.jsonl")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def salvar_log(dado: dict):
    dado["timestamp"] = datetime.now().isoformat()

    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(dado, ensure_ascii=False) + "\n")