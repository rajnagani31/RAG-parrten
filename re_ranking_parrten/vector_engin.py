from pydoc import text
from qdrant_client import QdrantClient
from qdrant_client.http import models
from openai import OpenAI
import os
from datetime import datetime
from dotenv import load_dotenv
import uuid # type: ignore 
import tiktoken
import fitz  # pymupdf




load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))




class QdrantVectorService:
    
    def __init__(self):
        self.collection_name = os.getenv("QDRANT_COLLECTION_NAME", "re_ranking_pattern")
        self.qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")

        self.client = QdrantClient(url=self.qdrant_url)
        self.embedding_model = "text-embedding-3-small"

        self.encoding = tiktoken.get_encoding("cl100k_base")

        self._create_collection()

    # =========================
    # EMBEDDING
    # =========================
    def create_embedding(self, text: str):
        response = client.embeddings.create(input=[text], model=self.embedding_model)
        return response.data[0].embedding

    # =========================
    # TEXT SPLITTING
    # =========================   
    def split_by_tokens(self, text: str, chunk_size=500, chunk_overlap=100):
        tokens = self.encoding.encode(text)
        chunks = []

        for i in range(0, len(tokens), chunk_size - chunk_overlap):
            chunk_tokens = tokens[i : i + chunk_size]
            chunk_text = self.encoding.decode(chunk_tokens)
            chunks.append(chunk_text)

        return chunks
    
    # =========================
    # COLLECTION
    # =========================    
    def _create_collection(self):
        collections = self.client.get_collections().collections
        names = [c.name for c in collections]

        if self.collection_name not in names:
            print("🚀 Creating new collection...")

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=1536, distance=models.Distance.COSINE
                ),
            )

    # =========================
    # STORE
    # =========================
    def store(self, user_id: int, text: str, type_: str):
        chunks = self.split_by_tokens(text)

        points = []

        for chunk in chunks:
            embedding = self.create_embedding(chunk)

            points.append(
                models.PointStruct(
                    id=str(uuid.uuid4()),  # Generate a unique ID for each point
                    vector=embedding,
                    payload={
                        "user_id": user_id,
                        "text": chunk,
                        "type": type_,
                        "timestamp": datetime.utcnow().isoformat(),
                    },
                )
            )

        self.client.upsert(collection_name=self.collection_name, points=points)

    # =========================
    # SEARCH
    # =========================
    def search(self, user_id: int, query: str, k=10):
        query_vector = self.create_embedding(query)

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=k,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="user_id", match=models.MatchValue(value=user_id)
                    )
                ]
            ),
        )

        # chunk for each query results

        result = []

        for rank,point in enumerate(response.points, start=1):
            result.append({
                "chunk": point.payload.get("text", ""), # type: ignore
                "score": point.score,
                "rank": rank,
                "query": query
            })

        return result


# store pdf vector for test re-ranking pattern

file_path = "nodejs.pdf"

import fitz  # pymupdf

def extract_pdf_text(pdf_path):
    doc = fitz.open(pdf_path)

    text = ""

    for page in doc:
        text += page.get_text() # type: ignore

    return text

if __name__ == "__main__":
    vector_service = QdrantVectorService()

    text = extract_pdf_text(file_path)

    vector_service.store(
        user_id=1,
        text=text,
        type_="pdf"
    )