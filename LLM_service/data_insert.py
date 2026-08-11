import os
import uuid
from pathlib import Path
from typing import List, Tuple

from langchain_text_splitters import TokenTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


# ================= КОНФИГУРАЦИЯ =================
KNOWLEDGE_BASE_DIR = "knowledge_base"
QDRANT_URL = "http://qdrant:6333"
QDRANT_COLLECTION_NAME = "rag_collection"

CHUNK_SIZE = 250      # в токенах
CHUNK_OVERLAP = 50    # в токенах


def load_and_split_with_metadata(folder: str) -> List[Tuple[str, dict]]:
    path = Path(folder)
    if not path.exists():
        raise FileNotFoundError(f"Папка {folder} не найдена")

    items = []  # Список кортежей: (текст чанка, метаданные)

    splitter = TokenTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    for file_path in path.glob("*.md"):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        # Разбиваем один файл на чанки
        file_chunks = splitter.split_text(text)
        # Для каждого чанка добавляем метаданные с именем файла
        for idx, chunk in enumerate(file_chunks):
            items.append((chunk, {"text": chunk, "source": file_path.name, "chunk_index": idx}))
    return items


def main():
    print("Загрузка и подготовка данных...")
    
    # 1. Загрузка, чанкинг и сбор метаданных
    items = load_and_split_with_metadata(KNOWLEDGE_BASE_DIR)
    
    if not items:
        print("Нет .md файлов или все файлы пустые.")
        return

    chunks = [item[0] for item in items]
    metadatas = [item[1] for item in items]

    print(f"Получено {len(chunks)} чанков из {len(set(m['source'] for m in metadatas))} файлов.")

    # 2. Модель и эмбеддинги
    print("Инициализация модели")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    print("Инициализация эмбеддингов")
    embeddings = model.encode(chunks)

    # 3. Qdrant
    print("Подключение к QDRANT")
    client = QdrantClient(url=QDRANT_URL)

    print("Создание коллекции")
    client.recreate_collection(
        collection_name=QDRANT_COLLECTION_NAME,
        vectors_config={"size": embeddings.shape[1], "distance": "Cosine"},
    )

    # 4. Сохранение в векторное хранилище (с метаданными)
    print("Сохранение чанков и метаданных в Qdrant...")

    
    points = []
    for (chunk, meta), vec in zip(items, embeddings):
        points.append(
            {
            "id": str(uuid.uuid4()),
            "vector": vec.tolist(),
            "payload": meta
        }
    )

    client.upsert(collection_name=QDRANT_COLLECTION_NAME, points=points)
    return print("Данные загружены в Qdrant.")

if __name__ == "__main__":
    main()
