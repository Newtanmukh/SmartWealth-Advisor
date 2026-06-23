%%writefile app.py
import io
import json
import os
import re
import shlex
import subprocess
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
import pandas as pd
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# -----------------------------------------------------------------------------
# Helpers and runtime utilities
# -----------------------------------------------------------------------------

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MODEL_CANDIDATES = [
    "llama3.2"
]
COMPLAINT_CATEGORIES = [
    "UPI",
    "ATM",
    "Debit Card",
    "Credit Card",
    "Internet Banking",
    "Mobile Banking",
    "Loan",
    "KYC",
    "Charges",
    "Account Opening",
    "Other",
]

import requests

def get_available_ollama_model() -> Optional[str]:
    try:
        response = requests.get(
            "http://localhost:11434/api/tags",
            timeout=10
        )

        response.raise_for_status()

        models = [
            m["name"]
            for m in response.json().get("models", [])
        ]

        if not models:
            return None

        return models[0]

    except Exception:
        return None

def run_ollama_generate(
    prompt: str,
    model_name: str,
    max_tokens: int = 512
) -> str:

    try:
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.2
            }
        }

        response = requests.post(
            "http://localhost:11434/api/generate",
            json=payload,
            timeout=600
        )

        response.raise_for_status()

        return response.json()["response"]

    except Exception as exc:
        raise RuntimeError(
            f"Ollama generation failed: {str(exc)}"
        )

@st.cache_resource
def load_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL_NAME)

def normalize_embeddings(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms

def initialize_session_state() -> None:
    if "knowledge_base" not in st.session_state:
        st.session_state.knowledge_base = {
            "documents": [],
            "chunks": [],
            "metadata": [],
            "index": None,
            "vector_count": 0,
            "model_name": None,
        }
    if "complaints" not in st.session_state:
        st.session_state.complaints = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "feedback" not in st.session_state:
        st.session_state.feedback = ""

def split_text(text: str, chunk_size: int = 300, chunk_overlap: int = 50) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= chunk_size:
        return [text]

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= chunk_size:
            current_chunk = f"{current_chunk} {sentence}".strip()
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk)

    if chunk_overlap > 0 and len(chunks) > 1:
        merged_chunks = []
        for i, chunk in enumerate(chunks):
            if i == 0:
                merged_chunks.append(chunk)
            else:
                overlap = " ".join(chunks[i - 1].split()[-chunk_overlap:])
                merged_chunks.append(f"{overlap} {chunk}".strip())
        chunks = merged_chunks

    return chunks

def extract_pdf_text(file_buffer: io.BytesIO) -> str:
    reader = PdfReader(file_buffer)
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages).strip()

def build_faiss_index(chunks: List[str]) -> Tuple[faiss.IndexFlatIP, np.ndarray]:
    embedder = load_embedding_model()
    embeddings = embedder.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    embeddings = normalize_embeddings(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index, embeddings

def update_knowledge_base(files: List[Any]) -> None:
    knowledge_base = st.session_state.knowledge_base
    all_chunks = []
    metadata = []
    documents = []

    for uploaded in files:
        try:
            text = extract_pdf_text(uploaded)
        except Exception as exc:
            st.error(f"Failed to read {uploaded.name}: {exc}")
            continue

        if not text:
            st.warning(f"Uploaded {uploaded.name} did not contain readable text.")
            continue

        documents.append(uploaded.name)
        chunks = split_text(text)
        for chunk_id, chunk_text in enumerate(chunks, start=1):
            all_chunks.append(chunk_text)
            metadata.append({
                "source": uploaded.name,
                "chunk_id": chunk_id,
                "text": chunk_text,
            })

    if not all_chunks:
        st.warning("No valid document text was processed.")
        return

    index, embeddings = build_faiss_index(all_chunks)
    knowledge_base["documents"] = documents
    knowledge_base["chunks"] = all_chunks
    knowledge_base["metadata"] = metadata
    knowledge_base["index"] = index
    knowledge_base["vector_count"] = len(all_chunks)
    st.success("Knowledge base updated with uploaded policy documents.")

def clear_knowledge_base() -> None:
    st.session_state.knowledge_base = {
        "documents": [],
        "chunks": [],
        "metadata": [],
        "index": None,
        "vector_count": 0,
        "model_name": st.session_state.knowledge_base.get("model_name"),
    }
    st.session_state.chat_history = []
    st.success("Knowledge base cleared.")

def is_kb_ready() -> bool:
    return st.session_state.knowledge_base.get("index") is not None and len(st.session_state.knowledge_base.get("chunks", [])) > 0

def retrieve_policy_chunks(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    if not is_kb_ready():
        return []

    embedder = load_embedding_model()
    query_vector = embedder.encode([query], convert_to_numpy=True)
    query_vector = normalize_embeddings(query_vector)
    index = st.session_state.knowledge_base["index"]
    scores, indices = index.search(query_vector, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(st.session_state.knowledge_base["metadata"]):
            continue
        item = st.session_state.knowledge_base["metadata"][idx]
        results.append({
            "source": item["source"],
            "text": item["text"],
            "score": float(score),
        })
    return results

def classify_complaint(complaint: str) -> str:
    normalized = complaint.lower()
    category_keywords = {
        "UPI": ["upi", "unified payment interface", "bhim", "peer to peer payment"],
        "ATM": ["atm", "cash dispenser", "withdraw", "not dispensing", "eject"],
        "Debit Card": ["debit card", "atm card", "card blocked", "card declined"],
        "Credit Card": ["credit card", "statement", "limit", "billing"],
        "Internet Banking": ["internet banking", "net banking", "login", "password", "say bank"],
        "Mobile Banking": ["mobile banking", "app", "mobile app", "OTP", "mPIN"],
        "Loan": ["loan", "EMI", "interest rate", "disbursement"],
        "KYC": ["kyc", "know your customer", "document", "address proof"],
        "Charges": ["charge", "fee", "deducted", "penalty"],
        "Account Opening": ["account opening", "new account", "account number", "savings account"],
    }
    for category, keywords in category_keywords.items():
        for keyword in keywords:
            if keyword in normalized:
                return category
    return "Other"

def generate_complaint_summary(complaint: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", complaint.strip())
    if len(sentences) <= 2:
        return complaint.strip()
    return " ".join(sentences[:2])

def detect_priority(complaint: str, category: str = "Other") -> str:
    normalized = complaint.lower()
    high_keywords = [
        "fraud", "unauthorized", "stolen", "hacked", "chargeback", "wrong debit",
        "money debited", "cash not dispensed", "cash missing", "failed transaction",
        "duplicate debit", "card blocked", "card stolen", "account taken over",
    ]
    medium_keywords = [
        "delay", "pending", "dispute", "complaint", "reversal", "refund",
        "kyc", "login issue", "otp", "password", "internet banking", "mobile banking",
    ]
    if any(keyword in normalized for keyword in high_keywords):
        return "High"
    if category in {"Loan", "KYC", "Credit Card", "Charges"} or any(keyword in normalized for keyword in medium_keywords):
        return "Medium"
    return "Low"

def detect_sentiment(complaint: str) -> str:
    normalized = complaint.lower()
    negative_words = [
        "angry", "frustrated", "worried", "irritated", "delay", "failed", "not working",
        "charged", "deducted", "blocked", "stolen", "unauthorized", "refused", "denied",
        "issue", "problem", "complaint", "not received", "no response",
    ]
    positive_words = ["thanks", "resolved", "satisfied", "good", "appreciate", "helpful"]
    score = sum(1 for word in negative_words if word in normalized) - sum(1 for word in positive_words if word in normalized)
    if score >= 2:
        return "Negative"
    if score <= -1:
        return "Positive"
    return "Neutral"

def complaints_to_dataframe(complaints: List[Dict[str, Any]]) -> pd.DataFrame:
    if not complaints:
        return pd.DataFrame(
            columns=[
                "complaint_id", "created_at", "category", "priority", "sentiment",
                "status", "summary", "resolution", "complaint_text"
            ]
        )
    return pd.DataFrame([
        {
            "complaint_id": c.get("complaint_id"),
            "created_at": c.get("created_at"),
            "category": c.get("category"),
            "priority": c.get("priority"),
            "sentiment": c.get("sentiment"),
            "status": c.get("status"),
            "summary": c.get("summary"),
            "resolution": c.get("resolution"),
            "complaint_text": c.get("complaint_text"),
        }
        for c in complaints
    ])

def export_complaints_csv(complaints: List[Dict[str, Any]]) -> bytes:
    df = complaints_to_dataframe(complaints)
    return df.to_csv(index=False).encode("utf-8")

def export_complaints_excel(complaints: List[Dict[str, Any]]) -> bytes:
    df = complaints_to_dataframe(complaints)
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Complaints")
    return buffer.getvalue()


def show_metrics_dashboard(complaints: List[Dict[str, Any]]) -> None:
    if not complaints:
        st.info("No complaint data available yet.")
        return

    df = complaints_to_dataframe(complaints)
    df["created_at_dt"] = pd.to_datetime(df["created_at"], errors="coerce")

    st.subheader("Complaint Summary")

    total = len(df)
    categories = df["category"].fillna("Unknown").value_counts()
    statuses = df["status"].fillna("Unknown").value_counts()
    priorities = df["priority"].fillna("Unknown").value_counts()
    sentiments = df["sentiment"].fillna("Unknown").value_counts()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Complaints", total)
    c2.metric("Unique Categories", int(categories.shape[0]))
    c3.metric("High Priority", int(priorities.get("High", 0)))
    c4.metric("Positive Sentiment", int(sentiments.get("Positive", 0)))

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Pending", int(statuses.get("Pending", 0)))
    c6.metric("Completed", int(statuses.get("Completed", 0)))
    c7.metric("Rejected", int(statuses.get("Rejected", 0)))
    c8.metric("Low Priority", int(priorities.get("Low", 0)))

    st.markdown("**Top Complaint Categories**")
    st.write(categories.head(10))

    st.markdown("**Complaint Status Counts**")
    st.write(statuses)

    st.markdown("**Priority Counts**")
    st.write(priorities.reindex(["High", "Medium", "Low"]).fillna(0).astype(int))

    st.markdown("**Sentiment Counts**")
    st.write(sentiments)

    valid_dates = df.dropna(subset=["created_at_dt"]).sort_values("created_at_dt")
    if not valid_dates.empty:
        daily_counts = valid_dates.groupby(valid_dates["created_at_dt"].dt.date).size()
        st.markdown("**Complaints Over Time (Counts by Date)**")
        st.write(daily_counts.tail(14))

        most_recent = valid_dates["created_at_dt"].max()
        last_7_days = valid_dates[valid_dates["created_at_dt"] >= (most_recent - pd.Timedelta(days=7))]
        st.metric("Complaints in Last 7 Days", int(len(last_7_days)))

        top_category = categories.index[0] if not categories.empty else "Unknown"
        st.metric("Most Frequent Category", str(top_category))


def parse_json_response(response: str) -> Dict[str, Any]:
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        # Attempt to extract JSON block from response text
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
    return {}

def compose_complaint_prompt(
    complaint: str, retrieved_chunks: List[Dict[str, Any]], category_hint: str
) -> str:
    chunks_text = "\n\n".join(
        [f"[{idx + 1}] Source: {chunk['source']}\n{chunk['text']}" for idx, chunk in enumerate(retrieved_chunks)]
    )
    prompt = (
        "You are a banking policy assistant. Based on the customer complaint and the policy excerpts below, respond in valid JSON. "
        "Use source numbers for citations."
        "\n\nComplaint:\n" + complaint + "\n\n"
        "Policy excerpts:\n" + chunks_text + "\n\n"
        "Produce an object with these keys:\n"
        "summary, category, suggested_resolution, next_actions, cited_sources.\n"
        "- summary: a short complaint summary.\n"
        "- category: one of UPI, ATM, Debit Card, Credit Card, Internet Banking, Mobile Banking, Loan, KYC, Charges, Account Opening, Other.\n"
        "- suggested_resolution: a concise policy-based resolution.\n"
        "- next_actions: actionable steps for the banking operations team. Return either a list of string steps or a string containing distinct steps.\n"
        "- cited_sources: a list of source reference numbers from the policy excerpts.\n\n"
        "If the complaint falls into one of the listed categories, use that category; otherwise use 'Other'.\n"
        "Category hint: "
        + category_hint
    )
    return prompt

def analyze_complaint(complaint: str) -> Dict[str, Any]:
    if not complaint.strip():
        raise ValueError("Complaint text cannot be empty.")
    if not is_kb_ready():
        raise RuntimeError("The knowledge base is not ready. Upload policies first.")

    category_hint = classify_complaint(complaint)
    summary = generate_complaint_summary(complaint)
    retrieved_chunks = retrieve_policy_chunks(complaint, top_k=3)
    if not retrieved_chunks:
        raise RuntimeError("No relevant policy chunks were retrieved from the knowledge base.")

    prompt = compose_complaint_prompt(complaint, retrieved_chunks, category_hint)
    model_name = st.session_state.knowledge_base.get("model_name") or "unknown"
    answer = run_ollama_generate(prompt, model_name, max_tokens=450)
    parsed = parse_json_response(answer)
    priority_hint = detect_priority(complaint, category_hint)
    sentiment_hint = detect_sentiment(complaint)
    result = {
        "summary": parsed.get("summary", summary),
        "category": parsed.get("category", category_hint),
        "priority": parsed.get("priority", priority_hint),
        "sentiment": parsed.get("sentiment", sentiment_hint),
        "suggested_resolution": parsed.get("suggested_resolution", "No resolution could be generated."),
        "next_actions": parsed.get("next_actions", "Review the policy excerpts and follow standard resolution steps."),
        "cited_sources": parsed.get("cited_sources", []),
        "retrieved_chunks": retrieved_chunks,
    }
    return result

def compose_chatbot_prompt(question: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
    chunks_text = "\n\n".join(
        [f"[{idx + 1}] Source: {chunk['source']}\n{chunk['text']}" for idx, chunk in enumerate(retrieved_chunks)]
    )
    prompt = (
        "You are a banking policy assistant. Answer the user question using only the policy excerpts below. "
        "If the answer is not in the excerpts, respond that you do not have enough information. "
        "Use source numbers in parentheses to cite relevant excerpts.\n\n"
        "Question:\n" + question + "\n\n"
        "Policy excerpts:\n" + chunks_text + "\n\n"
        "Provide a clear answer and cite sources like (1), (2)."
    )
    return prompt

def add_chat_history(question: str, answer: str, sources: List[str]) -> None:
    st.session_state.chat_history.append(
        {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "question": question,
            "answer": answer,
            "sources": sources,
        }
    )

# -----------------------------------------------------------------------------
# UI sections
# -----------------------------------------------------------------------------

def render_header() -> None:
    st.title("ComplaintIQ — AI-Powered Banking Complaint Resolution Assistant")
    st.markdown(
        "Streamlit app for local RAG-powered policy retrieval, complaint analytics, and interactive complaint tracking. "
        "Uses FAISS, sentence-transformers embeddings, and a local Ollama model."
    )

def tab_knowledge_base() -> None:
    st.header("Knowledge Base Management")
    st.markdown(
        "Upload banking policy or complaint resolution PDFs here. The app will extract text, create embeddings, and build a local FAISS knowledge base."
    )

    with st.expander("Upload new policy documents"):
        uploaded_files = st.file_uploader(
            "Select PDF files", type=["pdf"], accept_multiple_files=True, help="Upload policy or procedure documents."
        )
        if st.button("Build / Update Knowledge Base"):
            if not uploaded_files:
                st.warning("Please upload at least one PDF document before building the knowledge base.")
            else:
                update_knowledge_base(uploaded_files)

    if st.button("Clear Knowledge Base"):
        clear_knowledge_base()

    knowledge_base = st.session_state.knowledge_base
    st.subheader("Knowledge Base Status")
    col1, col2, col3 = st.columns(3)
    col1.metric("Uploaded Documents", len(knowledge_base["documents"]))
    col2.metric("Chunks Created", len(knowledge_base["chunks"]))
    col3.metric("Vectors Stored", knowledge_base["vector_count"])

    if knowledge_base["documents"]:
        st.subheader("Uploaded Documents")
        st.write(
            pd.DataFrame(
                {"Document Name": knowledge_base["documents"]}
            )
        )
    else:
        st.info("No policy documents have been uploaded yet.")

    model_name = knowledge_base.get("model_name")
    st.sidebar.markdown("### Local Model Status")
    if model_name:
        st.sidebar.success(f"Using Ollama model: {model_name}")
    else:
        st.sidebar.warning("Ollama model not detected yet. Please restart after installing Ollama and a local model.")

def write_complaint_record(record: Dict[str, Any]) -> None:
    st.session_state.complaints.append(record)

def display_next_actions_as_bullets(next_actions: Any) -> None:
    if isinstance(next_actions, list):
        for action in next_actions:
            st.markdown(f"- {action}")
    elif isinstance(next_actions, str):
        actions = [a.strip().lstrip("-*• ").strip() for a in re.split(r'(?:\r?\n)+', next_actions) if a.strip()]
        if len(actions) > 1:
            for action in actions:
                st.markdown(f"- {action}")
        else:
            st.markdown(f"- {next_actions}")
    else:
        st.markdown(f"- {str(next_actions)}")

def tab_complaint_analysis() -> None:
    st.header("Complaint Analysis")
    st.markdown(
        "Enter a customer complaint in the text box below. The app will analyze it, retrieve relevant policy excerpts, and generate a suggested policy-based resolution."
    )

    complaint_text = st.text_area(
        "Customer complaint text", height=240, placeholder="Enter the full complaint here..."
    )
    analyze_button = st.button("Analyze Complaint")

    if analyze_button:
        if not complaint_text.strip():
            st.warning("Please enter a complaint before clicking Analyze Complaint.")
        elif not is_kb_ready():
            st.warning("Knowledge base is empty. Please upload policy documents in the Knowledge Base tab first.")
        else:
            with st.spinner("Analyzing complaint and generating suggested resolution..."):
                try:
                    result = analyze_complaint(complaint_text)
                    complaint_id = len(st.session_state.complaints) + 1
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    record = {
                        "complaint_id": complaint_id,
                        "complaint_text": complaint_text.strip(),
                        "summary": result["summary"],
                        "category": result["category"],
                        "priority": result["priority"],
                        "sentiment": result["sentiment"],
                        "resolution": result["suggested_resolution"],
                        "next_actions": result["next_actions"],
                        "cited_sources": result["cited_sources"],
                        "status": "Pending",
                        "created_at": timestamp,
                        "retrieved_chunks": result["retrieved_chunks"],
                    }
                    write_complaint_record(record)

                    st.success("Complaint analyzed and stored successfully.")
                    st.subheader("Analysis Result")
                    st.markdown(f"**Complaint Summary:** {record['summary']}")
                    st.markdown(f"**Complaint Category:** {record['category']}")
                    st.markdown(f"**Priority:** {record['priority']}")
                    st.markdown(f"**Sentiment:** {record['sentiment']}")
                    st.markdown(f"**Suggested Resolution:** {record['resolution']}")
                    st.markdown("**Recommended Next Actions:**")
                    display_next_actions_as_bullets(record["next_actions"])
                    if record["cited_sources"]:
                        st.markdown(
                            "**Cited Source References:** "
                            + ", ".join([f"Excerpt {src}" for src in record["cited_sources"]])
                        )
                    st.markdown("---")
                    st.subheader("Retrieved Policy Snippets")
                    for idx, chunk in enumerate(result["retrieved_chunks"], start=1):
                        st.write(f"**Excerpt {idx} — {chunk['source']}**")
                        st.write(chunk["text"])

                except Exception as exc:
                    st.error(f"Analysis failed: {exc}")

    if st.session_state.complaints:
        st.info("Past analyzed complaints are available in the Resolution Tracker tab.")

def update_complaint_status(complaint_id: int, new_status: str) -> None:
    for complaint in st.session_state.complaints:
        if complaint["complaint_id"] == complaint_id:
            complaint["status"] = new_status
            st.success(f"Complaint {complaint_id} updated to {new_status}.")
            return
    st.error(f"Complaint {complaint_id} not found.")

def delete_complaint(complaint_id: int) -> None:
    st.session_state.complaints = [
        c for c in st.session_state.complaints if c["complaint_id"] != complaint_id
    ]
    st.success(f"Complaint {complaint_id} deleted.")

def status_color(status: str) -> str:
    if status == "Pending":
        return "#F6C23E"
    if status == "Completed":
        return "#1CC88A"
    if status == "Rejected":
        return "#E74A3B"
    return "#6C757D"

def tab_resolution_tracker() -> None:
    st.header("Resolution Tracker")
    st.markdown("Track complaint progress, update statuses, and review analytics for all analyzed complaints.")

    complaints = st.session_state.complaints
    total = len(complaints)
    pending = sum(1 for c in complaints if c["status"] == "Pending")
    completed = sum(1 for c in complaints if c["status"] == "Completed")
    rejected = sum(1 for c in complaints if c["status"] == "Rejected")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Complaints", total)
    col2.metric("Pending", pending)
    col3.metric("Completed", completed)
    col4.metric("Rejected", rejected)

    if total == 0:
        st.info("No complaints have been analyzed yet. Use the Complaint Analysis tab to add complaints.")
        return

    export_df = complaints_to_dataframe(complaints)
    csv_bytes = export_complaints_csv(complaints)
    xlsx_bytes = export_complaints_excel(complaints)

    st.subheader("Export Report")
    exp1, exp2 = st.columns(2)
    exp1.download_button(
        "Download CSV Report",
        data=csv_bytes,
        file_name="complaint_report.csv",
        mime="text/csv",
    )
    exp2.download_button(
        "Download Excel Report",
        data=xlsx_bytes,
        file_name="complaint_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.subheader("Complaint Category Breakdown")
    category_counts = pd.Series([c["category"] for c in complaints]).value_counts()
    for cat, count in category_counts.items():
        st.markdown(f"- **{cat} Complaints :** {count}")

    st.subheader("Complaint List")
    for complaint in complaints:
        card_color = status_color(complaint["status"])
        with st.container():
            st.markdown(
                f"<div style='border:1px solid #ddd; padding: 12px; border-radius: 10px; background: #FFF;'>"
                f"<strong>ID #{complaint['complaint_id']} — {complaint['category']}</strong><br>"
                f"<span style='color:{card_color}; font-weight:700'>{complaint['status']}</span><br>"
                f"<strong>Priority:</strong> {complaint.get('priority', 'Low')}<br>"
                f"<strong>Sentiment:</strong> {complaint.get('sentiment', 'Neutral')}<br>"
                f"<em>{complaint['summary']}</em><br>"
                f"<strong>Created:</strong> {complaint['created_at']}"
                f"</div>", unsafe_allow_html=True
            )
            cols = st.columns(4)
            if cols[0].button("Mark Completed", key=f"complete_{complaint['complaint_id']}"):
                update_complaint_status(complaint["complaint_id"], "Completed")
            if cols[1].button("Mark Pending", key=f"pending_{complaint['complaint_id']}"):
                update_complaint_status(complaint["complaint_id"], "Pending")
            if cols[2].button("Mark Rejected", key=f"reject_{complaint['complaint_id']}"):
                update_complaint_status(complaint["complaint_id"], "Rejected")
            if cols[3].button("Delete", key=f"delete_{complaint['complaint_id']}"):
                delete_complaint(complaint["complaint_id"])

            with st.expander("View full complaint details"):
                st.markdown(f"**Full Complaint:** {complaint['complaint_text']}")
                st.markdown(f"**Priority:** {complaint.get('priority', 'Low')}")
                st.markdown(f"**Sentiment:** {complaint.get('sentiment', 'Neutral')}")
                st.markdown(f"**Suggested Resolution:** {complaint['resolution']}")
                st.markdown("**Next Actions:**")
                display_next_actions_as_bullets(complaint["next_actions"])
                if complaint["cited_sources"]:
                    st.markdown(
                        "**Cited Source References:** "
                        + ", ".join([f"Excerpt {src}" for src in complaint["cited_sources"]])
                    )
                st.markdown("---")

def tab_ai_chatbot() -> None:
    st.header("AI Policy Chatbot")
    st.markdown("Ask policy-related questions and get answers grounded in the uploaded banking documents.")

    question = st.text_input("Policy question", placeholder="What is the SOP for failed UPI transactions?")
    ask_button = st.button("Ask Chatbot")

    if ask_button:
        if not question.strip():
            st.warning("Please enter a question for the chatbot.")
        elif not is_kb_ready():
            st.warning("Knowledge base is empty. Upload policy documents first.")
        else:
            with st.spinner("Retrieving policy excerpts and generating an answer..."):
                try:
                    retrieved_chunks = retrieve_policy_chunks(question, top_k=3)
                    if not retrieved_chunks:
                        st.warning("No relevant policy excerpts were found for this question.")
                    else:
                        prompt = compose_chatbot_prompt(question, retrieved_chunks)
                        model_name = st.session_state.knowledge_base.get("model_name") or "unknown"
                        answer = run_ollama_generate(prompt, model_name, max_tokens=350)
                        source_ids = [str(i + 1) for i in range(len(retrieved_chunks))]
                        add_chat_history(question, answer, source_ids)
                        st.subheader("Chatbot Answer")
                        st.write(answer)
                        st.markdown("**Cited Sources:** " + ", ".join([f"Excerpt {s}" for s in source_ids]))
                        with st.expander("Retrieved Policy Excerpts"):
                            for idx, chunk in enumerate(retrieved_chunks, start=1):
                                st.write(f"**Excerpt {idx} — {chunk['source']}**")
                                st.write(chunk["text"])
                except Exception as exc:
                    st.error(f"Chatbot failed: {exc}")

    if st.session_state.chat_history:
        st.subheader("Chat History")
        for entry in reversed(st.session_state.chat_history[-8:]):
            st.markdown(f"**{entry['timestamp']} — Question:** {entry['question']}")
            st.write(entry["answer"])
            st.markdown("**Sources:** " + ", ".join([f"Excerpt {s}" for s in entry["sources"]]))
            st.markdown("---")

def tab_analytics_dashboard() -> None:
    st.header("Analytics Dashboard")
    st.markdown("A higher-level view of complaint volume, priority, sentiment, and trends over time.")

    complaints = st.session_state.complaints
    if not complaints:
        st.info("No complaint data available yet. Analyze a few complaints first.")
        return

    total = len(complaints)
    high = sum(1 for c in complaints if c.get("priority") == "High")
    medium = sum(1 for c in complaints if c.get("priority") == "Medium")
    low = sum(1 for c in complaints if c.get("priority") == "Low")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Complaints", total)
    col2.metric("High Priority", high)
    col3.metric("Medium Priority", medium)
    col4.metric("Low Priority", low)

    show_metrics_dashboard(complaints)

def main() -> None:
    st.set_page_config(page_title="ComplaintIQ", page_icon="💼", layout="wide")
    initialize_session_state()

    if st.session_state.knowledge_base.get("model_name") is None:
        model_name = get_available_ollama_model()
        st.session_state.knowledge_base["model_name"] = model_name

    render_header()

    tab = st.sidebar.radio(
        "Navigation",
        [
            "Knowledge Base Management",
            "Complaint Analysis",
            "Resolution Tracker",
            "Analytics Dashboard",
            "AI Policy Chatbot",
        ],
    )

    if tab == "Knowledge Base Management":
        tab_knowledge_base()
    elif tab == "Complaint Analysis":
        tab_complaint_analysis()
    elif tab == "Resolution Tracker":
        tab_resolution_tracker()
    elif tab == "Analytics Dashboard":
        tab_analytics_dashboard()
    elif tab == "AI Policy Chatbot":
        tab_ai_chatbot()

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "Built with local LLMs, FAISS, sentence-transformers, and Streamlit. "
        "No cloud APIs are used."
    )

if __name__ == "__main__":
    main()
