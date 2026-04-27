"""
logger_metrics.py
=================
Salva cada consulta em JSONL para análise posterior no dashboard.

Formato de cada linha:
  {
    "timestamp": "2024-01-01T12:00:00",
    "query": "...",
    "response": "...",
    "chunks": [...],
    "latency_ms": 1200,
    "tokens_prompt": 800,
    "tokens_response": 300,
    "model": "llama-3.3-70b-versatile",
    "temperature": 0.1,
    "fallback": 0,
    "faithfulness": 0.85,      ← agora é float 0.0-1.0
    "citation_accuracy": 1.0,  ← agora é float 0.0-1.0
  }
"""

import json
from datetime import datetime
from pathlib import Path

LOG_PATH = Path("data/logs/logs.jsonl")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def salvar_log(dado: dict) -> None:
    dado["timestamp"] = datetime.now().isoformat()
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(dado, ensure_ascii=False) + "\n")