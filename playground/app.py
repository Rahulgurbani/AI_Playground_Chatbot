
import streamlit as st
import requests
import io
import time


BACKEND_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="AI Playground", layout="wide")
st.title("🧠 AI Playground Dashboard")


st.sidebar.header("⚙️ Configuration")
embedding = st.sidebar.selectbox("Embedding model", ["all-MiniLM-L6-v2", "BGE (bge-base-en-v1.5)"])
vectordb = st.sidebar.selectbox("Vector Database", ["weaviate", "redis"])
llm = st.sidebar.selectbox("LLM", ["gpt-j", "llama-3"])


st.subheader("📂 Upload and Ingest Document")

uploaded_file = st.file_uploader("Upload a text file (.txt only for now)", type=["txt"])

if uploaded_file is not None:
    file_content = uploaded_file.read().decode("utf-8")
    st.text_area("Preview", file_content[:1000], height=200)  # show first 1000 chars safely

    if st.button("📥 Ingest Document"):
        paragraphs = [p.strip() for p in file_content.split("\n\n") if p.strip()]
        total = len(paragraphs)
        success_count, fail_count = 0, 0

        progress_bar = st.progress(0)
        start_time = time.time()

        for i, para in enumerate(paragraphs):
            doc_id = f"upload_{i}"
            try:
                resp = requests.post(
                    f"{BACKEND_URL}/store",
                    params={"document_id": doc_id, "text": para},
                    timeout=60  
                )
                if resp.status_code == 200:
                    success_count += 1
                else:
                    fail_count += 1
                    st.warning(f"⚠️ Paragraph {i} failed with status {resp.status_code}: {resp.text}")
            except requests.exceptions.ConnectTimeout:
                st.error(f"⏱️ Timeout connecting to backend while processing paragraph {i}.")
                fail_count += 1
            except requests.exceptions.ConnectionError:
                st.error(f"❌ Could not reach backend (paragraph {i}). Is FastAPI running on port 8000?")
                fail_count += 1
                break
            except Exception as e:
                st.error(f"❌ Unexpected error on paragraph {i}: {e}")
                fail_count += 1

            progress_bar.progress((i + 1) / total)

        elapsed = time.time() - start_time
        st.success(f"✅ Successfully ingested {success_count}/{total} paragraphs into vector store in {elapsed:.2f}s!")
        if fail_count > 0:
            st.warning(f"⚠️ {fail_count} paragraphs failed to ingest. Check logs for details.")

st.divider()


st.subheader("💬 Chat Playground")

query = st.text_area("Enter your query", height=120, placeholder="Ask something about your documents...")

if st.button("🚀 Send"):
    if not query.strip():
        st.warning("Please enter a query before sending.")
    else:
        payload = {
            "config": {
                "embedding": embedding,
                "vectordb": vectordb,
                "llm": llm
            },
            "query": query.strip()
        }
        with st.spinner("💭 Contacting backend and retrieving relevant context..."):
            try:
                resp = requests.post(f"{BACKEND_URL}/chat", json=payload, timeout=60)
                if resp.status_code != 200:
                    st.error(f"Backend error ({resp.status_code}): {resp.text}")
                else:
                    data = resp.json()

                    
                    st.markdown("### 🧩 Response")
                    st.write(data.get("response", "No response received."))

                    
                    st.markdown("### 📄 Retrieved Documents")
                    retrieved = data.get("retrieved", [])
                    if retrieved:
                        for doc in retrieved:
                            with st.expander(f"Document ID: {doc.get('id', 'N/A')} (Score: {doc.get('score', 'N/A')})"):
                                st.write(doc.get("text", "No text found."))
                    else:
                        st.info("No relevant documents retrieved.")
            except requests.exceptions.ReadTimeout:
                st.error("⏱️ Backend took too long to respond. Try again with a shorter document or question.")
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to backend at 127.0.0.1:8000. Please start FastAPI first.")
            except Exception as e:
                st.error(f"❌ Unexpected error: {e}")
