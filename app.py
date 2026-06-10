"""
ComplaintIQ – AI-Powered Banking Complaint Resolution Assistant
Main Streamlit application entry point.
"""

import streamlit as st

# Page config — must be first Streamlit call
st.set_page_config(
    page_title="ComplaintIQ",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

from tabs import knowledge_base, complaint_analysis, resolution_tracker, policy_chatbot
from utils.session import init_session_state
from utils.styles import apply_styles

# Initialise session state
init_session_state()

# Inject custom CSS
apply_styles()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style='text-align:center; padding: 1rem 0 0.5rem;'>
            <span style='font-size:2.4rem;'>🏦</span>
            <h2 style='margin:0.2rem 0 0; letter-spacing:-0.5px;'>ComplaintIQ</h2>
            <p style='color:var(--muted); font-size:0.78rem; margin:0;'>
                AI-Powered Banking Complaint Resolution
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    # Knowledge base status
    kb = st.session_state.get("kb_stats", {})
    total_docs   = kb.get("total_docs", 0)
    total_chunks = kb.get("total_chunks", 0)

    status_color = "#22c55e" if total_docs > 0 else "#f59e0b"
    status_text  = "Ready" if total_docs > 0 else "No documents loaded"

    st.markdown(
        f"""
        <div style='background:var(--card-bg); border-radius:10px; padding:0.8rem 1rem; margin-bottom:0.5rem;'>
            <div style='display:flex; align-items:center; gap:0.5rem;'>
                <span style='color:{status_color}; font-size:0.7rem;'>●</span>
                <span style='font-size:0.78rem; font-weight:600;'>Knowledge Base</span>
            </div>
            <div style='font-size:0.72rem; color:var(--muted); margin-top:0.3rem;'>
                {status_text}<br/>
                {total_docs} doc{"s" if total_docs != 1 else ""} · {total_chunks} chunks
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Complaint summary
    complaints = st.session_state.get("complaints", [])
    pending    = sum(1 for c in complaints if c["status"] == "Pending")
    completed  = sum(1 for c in complaints if c["status"] == "Completed")
    rejected   = sum(1 for c in complaints if c["status"] == "Rejected")

    st.markdown(
        f"""
        <div style='background:var(--card-bg); border-radius:10px; padding:0.8rem 1rem;'>
            <div style='font-size:0.78rem; font-weight:600; margin-bottom:0.4rem;'>Complaints</div>
            <div style='display:flex; justify-content:space-between; font-size:0.72rem;'>
                <span>Total</span><span style='font-weight:600;'>{len(complaints)}</span>
            </div>
            <div style='display:flex; justify-content:space-between; font-size:0.72rem;'>
                <span style='color:#f59e0b;'>Pending</span><span style='color:#f59e0b; font-weight:600;'>{pending}</span>
            </div>
            <div style='display:flex; justify-content:space-between; font-size:0.72rem;'>
                <span style='color:#22c55e;'>Completed</span><span style='color:#22c55e; font-weight:600;'>{completed}</span>
            </div>
            <div style='display:flex; justify-content:space-between; font-size:0.72rem;'>
                <span style='color:#ef4444;'>Rejected</span><span style='color:#ef4444; font-weight:600;'>{rejected}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # Ollama model selector
    st.markdown("**⚙️ LLM Settings**")
    model = st.selectbox(
        "Ollama model",
        options=["qwen3:8b", "qwen2.5:7b", "llama3.2:3b", "mistral:7b", "phi3:mini"],
        index=0,
        key="ollama_model",
        help="Ensure this model is pulled in Ollama before use.",
    )
    st.caption("Run: `ollama pull qwen3:8b`")

    st.markdown(
        "<div style='font-size:0.7rem; color:var(--muted); margin-top:1rem;'>"
        "All data is in-session only.<br/>Data resets on page reload."
        "</div>",
        unsafe_allow_html=True,
    )

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📚 Knowledge Base",
        "🔍 Complaint Analysis",
        "📋 Resolution Tracker",
        "💬 Policy Chatbot",
    ]
)

with tab1:
    knowledge_base.render()

with tab2:
    complaint_analysis.render()

with tab3:
    resolution_tracker.render()

with tab4:
    policy_chatbot.render()