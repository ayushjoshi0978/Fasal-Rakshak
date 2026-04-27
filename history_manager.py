import json, os
from datetime import datetime

HISTORY_FILE = os.path.join(os.path.dirname(__file__), "..", "history", "scans.json")

def _ensure():
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)

def save_to_history(record: dict):
    _ensure()
    history = load_history()
    record["date"] = datetime.now().strftime("%d %b %Y, %I:%M %p")
    history.append(record)
    history = history[-100:]
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def load_history() -> list:
    _ensure()
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []
