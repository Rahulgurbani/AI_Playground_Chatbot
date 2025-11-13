
from typing import List, Dict, Any
import numpy as np


try:
    import weaviate
    from weaviate.classes.config import Property, DataType
except ImportError:
    weaviate = None


_in_memory_store: Dict[str, Dict[str, Any]] = {}


def get_weaviate_client():
    """
    Try connecting to local Weaviate instance.
    Returns client or None if connection fails.
    """
    if weaviate is None:
        print("⚠️ Weaviate module not installed — using in-memory mode.")
        return None

    try:
        client = weaviate.connect_to_local(
            host="localhost",
            port=8080,
            grpc_port=50051
        )
        print("✅ Connected to Weaviate (v4 client)")
        return client
    except Exception as e:
        print(f"❌ Could not connect to Weaviate: {e}")
        return None



def ensure_weaviate_schema(client):
    """
    Ensure 'Document' collection exists in Weaviate.
    """
    try:
        if client is None:
            return

        collections = [c.name for c in client.collections.list_all()]
        if "Document" not in collections:
            client.collections.create(
                name="Document",
                properties=[
                    Property(name="doc_id", data_type=DataType.TEXT),
                    Property(name="text", data_type=DataType.TEXT)
                ],
                vectorizer_config=None  # We provide our own embeddings
            )
            print("✅ Created collection: Document")
        else:
            print("📘 Collection 'Document' already exists.")
    except Exception as e:
        print(f"❌ Schema creation error: {e}")



def store_doc(doc_id: str, text: str, embedding: List[float], db_name: str = "weaviate") -> bool:
    """
    Store a text and its embedding in Weaviate or memory fallback.
    """
    try:
        if db_name.lower() == "weaviate":
            client = get_weaviate_client()
            if client:
                ensure_weaviate_schema(client)
                collection = client.collections.get("Document")

                collection.data.insert(
                    properties={
                        "doc_id": doc_id,
                        "text": text
                    },
                    vector=embedding
                )
                print(f"✅ Stored in Weaviate: {doc_id}")
                client.close()
                return True

        
        _in_memory_store[doc_id] = {"text": text, "embedding": embedding}
        print(f"⚠️ Stored in memory: {doc_id}")
        return True

    except Exception as e:
        print(f"❌ Error storing document: {e}")
        _in_memory_store[doc_id] = {"text": text, "embedding": embedding}
        return False



def query_docs(query_embedding: List[float], db_name: str = "weaviate", top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Retrieve top-k similar documents by cosine similarity.
    """
    try:
        if db_name.lower() == "weaviate":
            client = get_weaviate_client()
            if client:
                ensure_weaviate_schema(client)
                collection = client.collections.get("Document")

                result = collection.query.near_vector(
                    near_vector=query_embedding,
                    limit=top_k,
                    return_properties=["doc_id", "text"]
                )

                docs = []
                for o in result.objects:
                    docs.append({
                        "id": o.properties.get("doc_id", "N/A"),
                        "text": o.properties.get("text", ""),
                        "score": getattr(o.metadata, "distance", None)
                    })

                print(f"🔍 Retrieved {len(docs)} docs from Weaviate.")
                client.close()
                return docs

        
        print("⚠️ Using in-memory retrieval.")
        results = []
        for doc_id, rec in _in_memory_store.items():
            stored_vec = np.array(rec["embedding"])
            score = float(np.dot(stored_vec, query_embedding) /
                          (np.linalg.norm(stored_vec) * np.linalg.norm(query_embedding)))
            results.append({
                "id": doc_id,
                "text": rec["text"],
                "score": score
            })

        results = sorted(results, key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    except Exception as e:
        print(f"❌ Query error: {e}")
        return []
