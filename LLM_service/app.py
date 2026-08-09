# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from llama_cpp import Llama

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
llm = Llama(model_path="/app/models/tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf", n_ctx=2048, n_threads=1)

class QueryRequest(BaseModel):
    question: str
    context: str | None = None

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate")
def generate(req: QueryRequest):
    logger.info("Request received: question=%s, context=%s", req.question, req.context)
    prompt = f"Answer the question: {req.question}\nIf you don't know the answer - say so and specify."
    if req.context:
        prompt = f"Using context: {req.context}\nAnswer the question in 1-2 sentences. Don't make anything up, only use the context. Give only precise answers. If you don't know the answer - say so. Question: {req.question}"
    output = llm(prompt, max_tokens=256, temperature=0.7, stop=["\n\n"])
    return {"answer": output["choices"][0]["text"].strip()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
