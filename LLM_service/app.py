import logging
import os
from typing import Optional

import requests
from fastapi import FastAPI
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL_NAME = "gemma3:1b"  # или любая другая, которую ты пуллишь

class QueryRequest(BaseModel):
    question: str
    context: Optional[str] = None

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/generate")
def generate(req: QueryRequest):
    logger.info("Request received: question=%s", req.question)

    prompt = req.question
    if req.context:
        prompt = f"Using context: {req.context}\nAnswer the question in 2-3 sentences. Don't make anything up, only use the context. Give only precise answers. If you don't know the answer - say so. Don't mention that you were provided with context like `according to the text` etc. Question: {req.question}"

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }
    logger.info("Request sent to: "f"{OLLAMA_BASE_URL}/api/generate")
    logger.info(payload)
    resp = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload)
    resp.raise_for_status()
    data = resp.json()

    # Ollama возвращает поле response с текстом ответа
    answer = data.get("response", "").strip()
    return {"answer": answer}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
