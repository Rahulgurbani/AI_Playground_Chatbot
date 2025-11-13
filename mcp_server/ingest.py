import os
from backend.app.embeddings import get_embedding
from backend.app.vector_db import store_doc

def ingest_text_file(file_path: str, embedding_model: str = "all-MiniLM-L6-v2", vector_db: str = "weaviate"):
    """
    Reads a .txt file, generates embeddings, and stores it in the vector DB.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        
        embedding = get_embedding(text, model_name=embedding_model)

        
        doc_id = os.path.basename(file_path).replace(" ", "_").replace(".txt", "")

        
        store_doc(doc_id=doc_id, text=text, embedding=embedding, db_name=vector_db)

        print(f"✅ Ingested and stored document: {doc_id}")

    except Exception as e:
        print(f"❌ Error during ingestion: {e}")
