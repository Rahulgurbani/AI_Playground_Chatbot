# 🧠 AI Playground Chatbot
A full-stack AI application combining **FastAPI**, **Streamlit**, **Weaviate Vector DB**, and **LLMs** like GPT-J and LLaMA for intelligent document-aware chat.

## 🚀 Features
- 📄 Upload & ingest documents
- 🔍 Document chunking + embeddings
- 🧠 Vector search (Weaviate)
- 💬 Chat with RAG (LLM + retrieved docs)
- ⚡ FastAPI backend + Streamlit frontend

## 📁 Project Structure
AI_Playground_Chatbot/
│── backend/
│   └── app/
│── playground/
│── requirements.txt
│── README.md

## 🔧 Installation Guide
1. Clone repo:
   git clone https://github.com/Rahulgurbani/AI_Playground_Chatbot.git
2. Create env:
   python -m venv .venv
3. Activate:
   .venv\Scripts\activate
4. Install:
   pip install -r requirements.txt

## 🖥️ Run Backend
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000

## 🖼️ Run Frontend
streamlit run playground/app.py

## 🤝 Contribute
Open issues or PRs.

## 📜 License
MIT License © 2025 Rahul Gurbani
