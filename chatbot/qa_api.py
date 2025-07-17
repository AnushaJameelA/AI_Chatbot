

import os
import time
import requests
from typing import Dict, Any

HF_MODEL = os.getenv("HF_MODEL", "deepset/roberta-base-squad2")
HF_TOKEN = os.getenv("HF_TOKEN")
API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

def _require_token():
    if not HF_TOKEN:
        raise RuntimeError("Hugging Face token missing. Set HF_TOKEN env var or in Streamlit secrets.")

def answer_question(question: str, context: str, max_retries: int = 3) -> Dict[str, Any]:
    """
    Calls Hugging Face QA model with retries. Returns dict with:
    {ok: bool, answer: str, score: float, error: str, status_code: int}
    """
    _require_token()
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {"question": question, "context": context}

    attempt = 0
    while attempt < max_retries:
        attempt += 1
        try:
            resp = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        except Exception as e:
            return {"ok": False, "error": f"Request failed: {e}", "status_code": None, "answer": "", "score": 0.0}

        if resp.status_code == 200:
            try:
                data = resp.json()
                return {
                    "ok": True,
                    "answer": data.get("answer", ""),
                    "score": data.get("score", 0.0),
                    "status_code": 200,
                    "error": None
                }
            except Exception as e:
                return {"ok": False, "error": f"Parse error: {e}", "status_code": 200, "answer": "", "score": 0.0}

        
        if resp.status_code in (429, 503):
            time.sleep(3 * attempt)
            continue

        try:
            data = resp.json()
            err_msg = data.get("error", str(data))
        except:
            err_msg = f"HTTP {resp.status_code}"
        return {"ok": False, "error": err_msg, "status_code": resp.status_code, "answer": "", "score": 0.0}

    return {"ok": False, "error": "Max retries exceeded", "status_code": None, "answer": "", "score": 0.0}