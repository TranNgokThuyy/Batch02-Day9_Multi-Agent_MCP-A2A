import streamlit as st

from src.supervisor_workers import supervisor_answer

st.set_page_config(page_title="Drug Law RAG Chatbot", layout="wide")
st.title("RAG Chatbot — Pháp luật ma túy & tin tức liên quan")
st.caption("Pipeline: Weaviate Semantic + BM25 → Rerank → PageIndex fallback → Answer with citations")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Nhập câu hỏi về pháp luật ma túy hoặc tin tức liên quan...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Đang truy xuất tài liệu và tạo câu trả lời..."):
            result = supervisor_answer(query, top_k=5, score_threshold=0.05)
            answer = result["answer"] if isinstance(result, dict) else str(result)
            sources = result.get("sources", []) if isinstance(result, dict) else []
        st.markdown(answer)

        with st.expander("Source documents đã dùng"):
            for i, s in enumerate(sources, start=1):
                st.markdown(f"### Source {i}")
                st.write("Score:", s.get("score"))
                st.json(s.get("metadata", {}))
                st.write(s.get("content", "")[:2000])

    st.session_state.messages.append({"role": "assistant", "content": answer})
