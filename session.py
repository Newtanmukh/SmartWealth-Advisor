"""
utils/session.py – Centralised Streamlit session-state initialisation.
"""

import streamlit as st


def init_session_state() -> None:
    """Initialise all session-state keys once per browser session."""

    defaults = {
        # Knowledge base
        "faiss_index":    None,   # faiss.IndexFlatL2
        "kb_chunks":      [],     # list[str]  — raw text chunks
        "kb_metadata":    [],     # list[dict] — {source, page, chunk_id}
        "kb_stats": {
            "total_docs":   0,
            "total_chunks": 0,
            "doc_names":    [],
        },

        # Complaints
        "complaints":     [],     # list[dict]
        "complaint_counter": 0,   # auto-increment ID

        # Chat history (Policy Chatbot tab)
        "chat_history":   [],     # list[{"role": "user"|"assistant", "content": str, "sources": list}]

        # Ollama model (default; also set via sidebar selectbox)
        "ollama_model": "qwen3:8b",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value