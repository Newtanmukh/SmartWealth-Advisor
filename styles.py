"""
utils/styles.py – Custom CSS injected into the Streamlit app.
"""

import streamlit as st


def apply_styles() -> None:
    st.markdown(
        """
        <style>
        /* ── Design tokens ──────────────────────────────────────────────── */
        :root {
            --primary:    #1e40af;   /* deep bank-blue */
            --primary-lt: #3b82f6;
            --accent:     #0ea5e9;
            --success:    #22c55e;
            --warning:    #f59e0b;
            --danger:     #ef4444;
            --card-bg:    #f8fafc;
            --border:     #e2e8f0;
            --muted:      #64748b;
            --text:       #0f172a;
        }

        /* ── Global ─────────────────────────────────────────────────────── */
        html, body, [class*="css"] {
            font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
        }

        /* ── Tab styling ────────────────────────────────────────────────── */
        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
            border-bottom: 2px solid var(--border);
        }
        .stTabs [data-baseweb="tab"] {
            height: 46px;
            padding: 0 20px;
            border-radius: 8px 8px 0 0;
            font-size: 0.85rem;
            font-weight: 500;
            background: transparent;
            border: none;
            color: var(--muted);
        }
        .stTabs [aria-selected="true"] {
            background: white;
            color: var(--primary);
            border-bottom: 2px solid var(--primary);
            font-weight: 600;
        }

        /* ── Metric cards ───────────────────────────────────────────────── */
        .metric-card {
            background: white;
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.1rem 1.3rem;
            text-align: center;
            box-shadow: 0 1px 4px rgba(0,0,0,0.05);
        }
        .metric-card .metric-value {
            font-size: 2rem;
            font-weight: 700;
            line-height: 1;
            color: var(--primary);
        }
        .metric-card .metric-label {
            font-size: 0.78rem;
            color: var(--muted);
            margin-top: 0.3rem;
            font-weight: 500;
        }

        /* ── Status badges ──────────────────────────────────────────────── */
        .badge-pending   { background:#fef3c7; color:#92400e; padding:3px 10px; border-radius:20px; font-size:0.73rem; font-weight:600; }
        .badge-completed { background:#dcfce7; color:#14532d; padding:3px 10px; border-radius:20px; font-size:0.73rem; font-weight:600; }
        .badge-rejected  { background:#fee2e2; color:#7f1d1d; padding:3px 10px; border-radius:20px; font-size:0.73rem; font-weight:600; }

        /* ── Complaint result card ──────────────────────────────────────── */
        .result-card {
            background: white;
            border: 1px solid var(--border);
            border-left: 4px solid var(--primary);
            border-radius: 10px;
            padding: 1.2rem 1.4rem;
            margin: 0.6rem 0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        }
        .result-card h4 {
            margin: 0 0 0.5rem;
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: var(--muted);
        }
        .result-card p {
            margin: 0;
            font-size: 0.9rem;
            color: var(--text);
            line-height: 1.6;
        }

        /* ── Source citation box ────────────────────────────────────────── */
        .source-box {
            background: #f0f9ff;
            border: 1px solid #bae6fd;
            border-radius: 8px;
            padding: 0.7rem 1rem;
            margin: 0.35rem 0;
            font-size: 0.8rem;
            color: #0c4a6e;
        }

        /* ── Chat bubbles ───────────────────────────────────────────────── */
        .chat-user {
            background: var(--primary);
            color: white;
            border-radius: 18px 18px 4px 18px;
            padding: 0.7rem 1rem;
            max-width: 78%;
            margin-left: auto;
            margin-bottom: 0.6rem;
            font-size: 0.88rem;
        }
        .chat-assistant {
            background: white;
            border: 1px solid var(--border);
            border-radius: 18px 18px 18px 4px;
            padding: 0.7rem 1rem;
            max-width: 90%;
            margin-bottom: 0.6rem;
            font-size: 0.88rem;
            color: var(--text);
        }

        /* ── Upload zone ────────────────────────────────────────────────── */
        [data-testid="stFileUploader"] {
            border: 2px dashed var(--border);
            border-radius: 10px;
            padding: 0.5rem;
        }

        /* ── Buttons ────────────────────────────────────────────────────── */
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, var(--primary), var(--primary-lt));
            border: none;
            border-radius: 8px;
            font-weight: 600;
            letter-spacing: 0.02em;
        }

        /* ── Sidebar ────────────────────────────────────────────────────── */
        [data-testid="stSidebar"] {
            background: #f0f4ff;
        }

        /* ── Section heading ────────────────────────────────────────────── */
        .section-heading {
            font-size: 1.15rem;
            font-weight: 700;
            color: var(--text);
            margin-bottom: 0.2rem;
        }
        .section-sub {
            font-size: 0.82rem;
            color: var(--muted);
            margin-bottom: 1rem;
        }

        /* ── Info / warning banners ─────────────────────────────────────── */
        .info-banner {
            background: #eff6ff;
            border: 1px solid #bfdbfe;
            border-radius: 8px;
            padding: 0.7rem 1rem;
            font-size: 0.82rem;
            color: #1e40af;
            margin-bottom: 0.8rem;
        }
        .warn-banner {
            background: #fffbeb;
            border: 1px solid #fde68a;
            border-radius: 8px;
            padding: 0.7rem 1rem;
            font-size: 0.82rem;
            color: #92400e;
            margin-bottom: 0.8rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )