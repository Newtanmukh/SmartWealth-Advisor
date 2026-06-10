"""
tabs/complaint_analysis.py – Tab 2: Complaint Analysis
"""

from __future__ import annotations

import uuid
from datetime import datetime

import streamlit as st

from utils.llm import analyze_complaint, is_ollama_running, COMPLAINT_CATEGORIES
from utils.rag_engine import retrieve_relevant_chunks

# ── Sample complaints for quick testing ───────────────────────────────────────
SAMPLE_COMPLAINTS = {
    "UPI Failed Transaction": (
        "Customer attempted a UPI transfer of ₹15,000 to a beneficiary. "
        "The amount was debited from the savings account but the beneficiary did not receive "
        "the money. Transaction reference: UPI/2024/78934. The issue has been unresolved for 3 days. "
        "Customer has already raised the issue with their bank but no refund or credit has been processed."
    ),
    "ATM Cash Not Dispensed": (
        "Customer tried to withdraw ₹10,000 from ATM ID: SBI001234 at Andheri East, Mumbai. "
        "The account was debited but cash was not dispensed. The ATM screen showed 'Transaction Approved' "
        "but no cash came out. This happened on 12th February 2024 at 3:45 PM. "
        "Customer wants immediate reversal of the deducted amount."
    ),
    "Debit Card Blocked Incorrectly": (
        "Customer's debit card ending in 4521 was blocked without any notification. "
        "Customer was unable to make online payments and POS transactions. "
        "The customer never received any OTP abuse notification or fraud alert. "
        "The customer wants the card unblocked and an explanation for the blocking."
    ),
    "Internet Banking Access Issue": (
        "Customer is unable to log in to internet banking for the past 5 days. "
        "The system shows 'Invalid credentials' even after password reset. "
        "Customer has tried on multiple browsers and devices. "
        "This is severely impacting business transactions and salary transfers."
    ),
}


def render() -> None:
    st.markdown(
        "<div class='section-heading'>🔍 Complaint Analysis</div>"
        "<div class='section-sub'>Enter complaint text to get AI-powered resolution suggestions.</div>",
        unsafe_allow_html=True,
    )

    # ── Guard: No knowledge base ───────────────────────────────────────────────
    if st.session_state.get("faiss_index") is None:
        st.markdown(
            "<div class='warn-banner'>⚠️ Knowledge base is empty. "
            "Please upload policy documents in the <strong>Knowledge Base</strong> tab first. "
            "The system will still attempt analysis, but resolutions may be generic.</div>",
            unsafe_allow_html=True,
        )

    # ── Guard: Ollama not running ─────────────────────────────────────────────
    if not is_ollama_running():
        st.markdown(
            "<div class='warn-banner'>⚠️ Ollama is not running. "
            "Start it with <code>ollama serve</code> and pull your model with "
            "<code>ollama pull qwen3:8b</code>.</div>",
            unsafe_allow_html=True,
        )

    # ── Left / Right layout ───────────────────────────────────────────────────
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown("#### Complaint Input")

        # Quick sample loader
        sample_key = st.selectbox(
            "Load a sample complaint (optional)",
            options=["— type your own —"] + list(SAMPLE_COMPLAINTS.keys()),
            key="sample_select",
        )

        default_text = ""
        if sample_key != "— type your own —":
            default_text = SAMPLE_COMPLAINTS[sample_key]

        complaint_text = st.text_area(
            "Complaint Text",
            value=st.session_state.get("_complaint_draft", default_text),
            height=220,
            placeholder=(
                "e.g. Customer attempted a UPI transfer. Amount was debited from the "
                "account but the beneficiary did not receive the money…"
            ),
            key="complaint_input",
            label_visibility="collapsed",
        )

        # Options row
        col_a, col_b = st.columns(2)
        with col_a:
            top_k = st.slider("Retrieve top-K chunks", 2, 10, 5, key="top_k")
        with col_b:
            model = st.session_state.get("ollama_model", "qwen3:8b")
            st.markdown(
                f"<div style='padding-top:1.6rem; font-size:0.8rem; color:#64748b;'>"
                f"Model: <strong>{model}</strong></div>",
                unsafe_allow_html=True,
            )

        analyze_btn = st.button(
            "🔍 Analyse Complaint",
            type="primary",
            use_container_width=True,
        )

        if analyze_btn:
            st.session_state["_trigger_analysis"] = True
            st.session_state["_complaint_draft"]  = complaint_text

    with right:
        st.markdown("#### Analysis Result")
        _render_result_placeholder()

    # ── Run analysis ──────────────────────────────────────────────────────────
    if st.session_state.get("_trigger_analysis"):
        st.session_state["_trigger_analysis"] = False
        text = st.session_state.get("_complaint_draft", "").strip()

        if not text:
            st.error("Please enter complaint text before analysing.")
            return

        _run_analysis(text, top_k=st.session_state.get("top_k", 5))


def _render_result_placeholder() -> None:
    last = st.session_state.get("_last_analysis")
    if not last:
        st.markdown(
            "<div style='border:1px dashed #e2e8f0; border-radius:10px; padding:2rem; "
            "text-align:center; color:#94a3b8; font-size:0.85rem;'>"
            "Analysis results will appear here after you submit a complaint."
            "</div>",
            unsafe_allow_html=True,
        )
        return

    result   = last["result"]
    chunks   = last["chunks"]
    cmp_id   = last["complaint_id"]

    # Summary
    st.markdown(
        f"<div class='result-card'><h4>Summary</h4><p>{result['summary']}</p></div>",
        unsafe_allow_html=True,
    )

    # Category + Confidence
    conf_pct = int(result.get("confidence", 0.5) * 100)
    conf_color = "#22c55e" if conf_pct >= 70 else "#f59e0b" if conf_pct >= 40 else "#ef4444"

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"<div class='result-card'><h4>Category</h4>"
            f"<p style='font-size:1rem; font-weight:700; color:#1e40af;'>{result['category']}</p></div>",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"<div class='result-card'><h4>Confidence</h4>"
            f"<p style='font-size:1rem; font-weight:700; color:{conf_color};'>{conf_pct}%</p></div>",
            unsafe_allow_html=True,
        )

    # Resolution
    st.markdown(
        f"<div class='result-card' style='border-left-color:#22c55e;'>"
        f"<h4>Suggested Resolution</h4><p>{result['resolution']}</p></div>",
        unsafe_allow_html=True,
    )

    # Next actions
    actions = result.get("next_actions", [])
    if actions:
        actions_html = "".join(
            f"<li style='margin:0.3rem 0; font-size:0.88rem;'>✅ {a}</li>" for a in actions
        )
        st.markdown(
            f"<div class='result-card' style='border-left-color:#0ea5e9;'>"
            f"<h4>Recommended Next Actions</h4><ul style='margin:0; padding-left:1.2rem;'>"
            f"{actions_html}</ul></div>",
            unsafe_allow_html=True,
        )

    # Sources
    if chunks:
        with st.expander(f"📎 {len(chunks)} Policy Sources Retrieved", expanded=False):
            for i, c in enumerate(chunks, 1):
                score_pct = int(c.get("score", 0) * 100)
                st.markdown(
                    f"<div class='source-box'><strong>Source {i}</strong> · "
                    f"{c['source']} · Page {c['page']} · Relevance: {score_pct}%<br/>"
                    f"<span style='opacity:0.85;'>{c['text'][:300]}…</span></div>",
                    unsafe_allow_html=True,
                )

    # Complaint saved notice
    st.success(f"✅ Complaint saved as **{cmp_id}** (Status: Pending)")


def _run_analysis(complaint_text: str, top_k: int = 5) -> None:
    model = st.session_state.get("ollama_model", "qwen3:8b")

    with st.spinner("🔍 Retrieving relevant policy chunks…"):
        chunks = retrieve_relevant_chunks(
            complaint_text,
            st.session_state.get("faiss_index"),
            st.session_state.get("kb_chunks", []),
            st.session_state.get("kb_metadata", []),
            top_k=top_k,
        )

    with st.spinner(f"🤖 Analysing with {model}…"):
        try:
            result = analyze_complaint(complaint_text, chunks, model=model)
        except RuntimeError as e:
            st.error(str(e))
            return
        except Exception as e:
            st.error(f"Unexpected error during analysis: {e}")
            return

    # Save complaint record
    st.session_state["complaint_counter"] += 1
    cmp_id = f"CMP-{st.session_state['complaint_counter']:04d}"
    record = {
        "complaint_id":  cmp_id,
        "complaint_text": complaint_text,
        "summary":       result.get("summary", ""),
        "category":      result.get("category", "Other"),
        "resolution":    result.get("resolution", ""),
        "next_actions":  result.get("next_actions", []),
        "confidence":    result.get("confidence", 0.5),
        "status":        "Pending",
        "created_at":    datetime.now().strftime("%Y-%m-%d %H:%M"),
        "sources":       [{"source": c["source"], "page": c["page"]} for c in chunks],
    }
    st.session_state["complaints"].append(record)

    # Stash for display
    st.session_state["_last_analysis"] = {
        "result": result,
        "chunks": chunks,
        "complaint_id": cmp_id,
    }
    st.rerun()