import os
from typing import List, Tuple

import requests
import streamlit as st

API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8001")
REQUEST_TIMEOUT = 60

st.set_page_config(page_title="AI Customer Support Chatbot", page_icon=":speech_balloon:", layout="wide")
st.title("AI Customer Support Chatbot")
st.caption("Upload PDFs, ask questions, and get source-cited answers.")

if "messages" not in st.session_state:
    st.session_state.messages: List[dict] = []


def backend_ok() -> bool:
    try:
        res = requests.get(f"{API_BASE}/health", timeout=8)
        return res.status_code == 200
    except requests.RequestException:
        return False


def get_status_count() -> int:
    try:
        res = requests.get(f"{API_BASE}/documents/status", timeout=8)
        if res.status_code == 200:
            return int(res.json().get("document_count", 0))
    except requests.RequestException:
        pass
    return 0


def history_for_api() -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    pending_user = None
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            pending_user = msg["content"]
        elif msg["role"] == "assistant" and pending_user is not None:
            pairs.append((pending_user, msg["content"]))
            pending_user = None
    return pairs


with st.sidebar:
    st.subheader("System")
    if backend_ok():
        st.success("API Online")
    else:
        st.error("API Offline")

    st.write(f"Indexed chunks: **{get_status_count()}**")
    st.divider()

    uploaded_file = st.file_uploader("Browse files", type=["pdf"])
    if st.button("Process & Index", use_container_width=True, type="primary"):
        if uploaded_file is None:
            st.warning("Please choose a PDF file first.")
        else:
            with st.spinner("Uploading and indexing document..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    res = requests.post(
                        f"{API_BASE}/documents/upload",
                        files=files,
                        timeout=REQUEST_TIMEOUT,
                    )
                    if res.status_code == 200:
                        data = res.json()
                        st.success(
                            f"Indexed `{data.get('filename', uploaded_file.name)}` | "
                            f"Pages: {data.get('pages_processed', 0)} | "
                            f"Chunks: {data.get('chunks_created', 0)}"
                        )
                    else:
                        detail = res.json().get("detail", res.text)
                        st.error(f"Upload failed: {detail}")
                except requests.RequestException as exc:
                    st.error(f"Could not reach backend: {exc}")


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            for source in msg["sources"]:
                st.caption(f"Source: {source.get('source', 'unknown')} | Page {source.get('page', '?')}")


prompt = st.chat_input("Ask about your uploaded documents...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                payload = {
                    "question": prompt,
                    "history": history_for_api(),
                }
                res = requests.post(f"{API_BASE}/chat/", json=payload, timeout=REQUEST_TIMEOUT)
                if res.status_code == 200:
                    data = res.json()
                    answer = data.get("answer", "No answer returned.")
                    sources = data.get("sources", [])
                    st.markdown(answer)
                    for source in sources:
                        st.caption(f"Source: {source.get('source', 'unknown')} | Page {source.get('page', '?')}")
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer, "sources": sources}
                    )
                else:
                    detail = res.json().get("detail", res.text)
                    err = f"Chat failed: {detail}"
                    st.error(err)
                    st.session_state.messages.append({"role": "assistant", "content": err, "sources": []})
            except requests.RequestException as exc:
                err = f"Could not reach backend: {exc}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err, "sources": []})
