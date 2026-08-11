import logging
import os
from typing import Optional

import requests
from fastapi import FastAPI
from pydantic import BaseModel

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

QDRANT_URL = "http://qdrant:6333"
QDRANT_COLLECTION_NAME = "rag_collection"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "gemma3:1b")

logger.info("Инициализация клиента Qdrant")
client = QdrantClient(url=QDRANT_URL)
logger.info("Инициализация all-MiniLM")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
logger.info("Успешная инициализация")

class QueryRequest(BaseModel):
    question: str
    context: Optional[str] = None

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/generate")
def generate(req: QueryRequest):
    logger.info("Request received: question=%s", req.question)
    embeddings = model.encode([req.question])[0]
    logger.info("свекторизовали=%s", embeddings)
    logger.info("Поиск ответа")

    resp = client.query_points(
    collection_name = QDRANT_COLLECTION_NAME,
    query = embeddings.tolist(),
    limit = 5
    )

    threshold = 0.45 #порог попадения в контекст
    context_src = [point.payload["text"] for point in resp.points if point.score >= threshold]
    context = "\n\n".join(context_src)
    context += ""
    logger.info("получен контекст=%s", context)
    
    prompt = os.getenv("PROMPT_BASE") + f"QUESTION:  {req.question}"
    if context:
        prompt = prompt + f"CONTEXT: {context}"
    logger.info(prompt)
    
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,      # минимум случайности
            "top_p": 0.1,           # узкий выбор токенов
            "num_predict": 256      # лимит токенов
        }
    }
    logger.info("Request sent to: "f"{OLLAMA_BASE_URL}/api/generate")

    resp = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload)
    resp.raise_for_status()
    data = resp.json()

    # Ollama возвращает поле response с текстом ответа
    answer = data.get("response", "").strip()
    return {"answer": answer}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
