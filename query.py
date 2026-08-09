from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

QDRANT_URL = "http://localhost:6333"
QDRANT_COLLECTION_NAME = "rag_collection"


# Пример поиска
query = input("\nВведите поисковый запрос: ").strip()

client = QdrantClient(url=QDRANT_URL)
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(query)

print("\nПоиск ответа...")

resp = client.query_points(
    collection_name = QDRANT_COLLECTION_NAME,
    query = embeddings.tolist(),
    limit = 5
)

print(resp)