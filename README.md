@'
# AI Playground (starter)

Minimal starter for AI Playground:
- FastAPI backend (stubs for embeddings, vector DB, LLM)
- Streamlit frontend for quick testing
- Simple MCP ingester script

Run backend: uvicorn backend.app.main:app --reload --port 8000  
Run frontend: streamlit run playground/app.py

This skeleton uses mock implementations; we will replace them step-by-step with real models (sentence-transformers, Weaviate/Redis clients, and real LLM runner).
'@ | Out-File -Encoding utf8 README.md
