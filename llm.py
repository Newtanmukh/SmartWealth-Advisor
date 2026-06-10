"""
utils/llm.py – Ollama-based local LLM inference helpers.
"""

from __future__ import annotations

import json
import re
from typing import List

import requests

OLLAMA_BASE = "http://localhost:11434"
OLLAMA_API  = f"{OLLAMA_BASE}/api/generate"
OLLAMA_CHAT = f"{OLLAMA_BASE}/api/chat"


# ── Health check ──────────────────────────────────────────────────────────────

def is_ollama_running() -> bool:
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def list_local_models() -> List[str]:
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        data = r.json()
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


# ── Core completion ───────────────────────────────────────────────────────────

def _ollama_generate(model: str, prompt: str, system: str = "") -> str:
    """
    Call Ollama /api/generate (non-streaming).
    Returns the response text or raises RuntimeError.
    """
    payload: dict = {
        "model":  model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_predict": 1024,
        },
    }
    if system:
        payload["system"] = system

    try:
        resp = requests.post(OLLAMA_API, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "").strip()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Cannot connect to Ollama. Make sure Ollama is running: `ollama serve`"
        )
    except requests.exceptions.Timeout:
        raise RuntimeError("Ollama request timed out. The model may still be loading.")
    except Exception as e:
        raise RuntimeError(f"Ollama error: {e}") from e


# ── Complaint Analysis ────────────────────────────────────────────────────────

COMPLAINT_CATEGORIES = [
    "UPI", "ATM", "Debit Card", "Credit Card", "Internet Banking",
    "Mobile Banking", "Loan", "KYC", "Charges", "Account Opening", "Other",
]


def analyze_complaint(
        complaint_text: str,
        context_chunks: List[dict],
        model: str = "qwen3:8b",
) -> dict:
    """
    Analyse a complaint using retrieved context.
    Returns structured dict with: summary, category, resolution,
    next_actions, confidence, cited_sources.
    """
    context = _format_context(context_chunks)
    categories_str = ", ".join(COMPLAINT_CATEGORIES)

    system = (
        "You are ComplaintIQ, an AI assistant for an Indian bank's complaint resolution team. "
        "Analyse complaints professionally and provide actionable resolutions based on bank policies. "
        "Always respond in valid JSON only – no markdown, no preamble."
    )

    prompt = f"""Analyse the following customer complaint using the policy context provided.

COMPLAINT:
{complaint_text}

RELEVANT POLICY CONTEXT:
{context}

Respond ONLY with a JSON object (no markdown) with these exact keys:
{{
  "summary": "One-sentence summary of the complaint",
  "category": "One of: {categories_str}",
  "resolution": "Detailed resolution steps based on policy context (3-5 sentences)",
  "next_actions": ["Action 1", "Action 2", "Action 3"],
  "confidence": <float between 0.0 and 1.0>,
  "reasoning": "Brief explanation of why this resolution was chosen"
}}"""

    raw = _ollama_generate(model, prompt, system)
    return _parse_json_response(raw, complaint_text)


def _parse_json_response(raw: str, complaint_text: str) -> dict:
    """Try to extract JSON from the LLM output; fall back to defaults."""
    # Strip potential <think>...</think> tags from reasoning models
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    # Try direct parse
    try:
        data = json.loads(raw)
        _validate_complaint_dict(data)
        return data
    except Exception:
        pass

    # Try extracting JSON block
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            _validate_complaint_dict(data)
            return data
        except Exception:
            pass

    # Ultimate fallback
    return {
        "summary":      complaint_text[:120] + "…" if len(complaint_text) > 120 else complaint_text,
        "category":     "Other",
        "resolution":   raw[:600] if raw else "Unable to generate resolution. Please check Ollama.",
        "next_actions": ["Review complaint manually", "Contact customer for details"],
        "confidence":   0.4,
        "reasoning":    "Automated parsing failed; raw LLM output shown.",
    }


def _validate_complaint_dict(data: dict) -> None:
    required = {"summary", "category", "resolution", "next_actions", "confidence"}
    if not required.issubset(data.keys()):
        raise ValueError("Missing required keys")
    if data["category"] not in COMPLAINT_CATEGORIES:
        data["category"] = "Other"


# ── Policy Chatbot ────────────────────────────────────────────────────────────

def answer_policy_question(
        question: str,
        context_chunks: List[dict],
        chat_history: List[dict],
        model: str = "qwen3:8b",
) -> str:
    """Generate a chatbot answer grounded in retrieved policy context."""
    context = _format_context(context_chunks)

    history_str = ""
    if chat_history:
        recent = chat_history[-6:]  # last 3 turns
        history_str = "\n".join(
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in recent
        )

    system = (
        "You are a knowledgeable banking policy assistant for an Indian bank. "
        "Answer questions accurately based only on the provided policy documents. "
        "If information is not in the context, say so clearly. "
        "Be concise, professional, and helpful."
    )

    prompt = f"""Use the following policy context to answer the user's question.

POLICY CONTEXT:
{context}

{"CONVERSATION HISTORY:" + chr(10) + history_str if history_str else ""}

USER QUESTION: {question}

Provide a clear, professional answer based on the policy context above."""

    return _ollama_generate(model, prompt, system)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _format_context(chunks: List[dict]) -> str:
    if not chunks:
        return "No policy context available."
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(
            f"[Source {i}: {c['source']}, Page {c['page']}]\n{c['text']}"
        )
    return "\n\n".join(parts)