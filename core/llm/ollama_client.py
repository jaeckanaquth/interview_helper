# core/llm/ollama_client.py

import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "llama3.1"  # or whatever you named the model in `ollama list`


def generate_answer(system_prompt: str, user_message: str) -> str:
    """
    Call local Ollama chat API with a system prompt + user message
    and return the raw text response.
    """
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "stream": False,
    }

    resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data["message"]["content"]
