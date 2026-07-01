import streamlit as st

def inject_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700&family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@400;500;600&display=swap');

    :root {
        --paper: #f5f1e8;
        --paper-raised: #fdfbf5;
        --ink: #1c1a17;
        --ink-soft: #5c574d;
        --line: #ddd6c4;
        --rust: #c1502e;
        --moss: #4a5d4e;
        --gold: #b8923d;
    }

    .stApp {
        background-color: var(--paper);
        background-image: radial-gradient(circle at 1px 1px, rgba(28,26,23,0.04) 1px, transparent 0);
        background-size: 24px 24px;
    }

    html, body, [class*="css"], p, span, div, label {
        font-family: 'Inter', sans-serif;
        color: var(--ink);
    }

    h1, h2, h3 {
        font-family: 'Fraunces', serif !important;
        color: var(--ink) !important;
        letter-spacing: -0.01em !important;
    }

    h1 { font-size: 2.6rem !important; font-weight: 600 !important; line-height: 1.1 !important; margin-bottom: 0.3rem !important; }
    h2 { font-size: 1.5rem !important; font-weight: 600 !important; margin-top: 2.2rem !important; padding-bottom: 0.5rem; border-bottom: 2px solid var(--ink); display: inline-block; }
    h3 { font-size: 1.15rem !important; font-weight: 500 !important; color: var(--ink-soft) !important; }

    p, .stMarkdown, li { font-family: 'Inter', sans-serif !important; color: var(--ink-soft) !important; line-height: 1.65 !important; }

    [data-testid="stMetricValue"] { font-family: 'IBM Plex Mono', monospace !important; font-weight: 600 !important; color: var(--ink) !important; font-size: 1.8rem !important; }
    [data-testid="stMetricLabel"] { font-family: 'Inter', sans-serif !important; text-transform: uppercase; font-size: 0.68rem !important; font-weight: 600; letter-spacing: 0.1em; color: var(--ink-soft) !important; }
    [data-testid="stMetricDelta"] { font-family: 'IBM Plex Mono', monospace !important; font-size: 0.85rem !important; }

    [data-testid="stMetric"] { background: var(--paper-raised); border: 1.5px solid var(--ink); border-radius: 2px; padding: 1rem 1.3rem; box-shadow: 4px 4px 0px var(--ink); }

    [data-testid="stSidebar"] { background: var(--paper-raised); border-right: 1.5px solid var(--ink); }
    [data-testid="stSidebar"] h1 { font-size: 1.6rem !important; border-bottom: none; }
    [data-testid="stSidebar"] hr { border-color: var(--line) !important; }

    /* Hide Streamlit's auto-generated multipage nav links at top of sidebar */
    [data-testid="stSidebarNav"] { display: none !important; }

    [data-testid="stSidebar"] [role="radiogroup"] label {
        font-family: 'Inter', sans-serif; font-weight: 500; font-size: 0.92rem;
        padding: 0.5rem 0.7rem; border-radius: 3px; border-left: 3px solid transparent;
        color: var(--ink) !important;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:hover { background: var(--paper); border-left: 3px solid var(--rust); }
    [data-testid="stSidebar"] [role="radiogroup"] label p { color: var(--ink) !important; }

    .stButton button {
        font-family: 'Inter', sans-serif !important; font-weight: 600 !important; font-size: 0.95rem !important;
        background: var(--rust) !important; color: var(--paper-raised) !important;
        border: 1.5px solid var(--ink) !important; border-radius: 2px !important;
        padding: 0.6rem 1.4rem !important; box-shadow: 3px 3px 0px var(--ink);
        text-transform: uppercase; letter-spacing: 0.04em;
    }
    .stButton button p { color: var(--paper-raised) !important; }
    .stButton button:hover { background: #a8431f !important; transform: translate(-2px, -2px); box-shadow: 5px 5px 0px var(--ink); }

    /* Force readable text inside Streamlit dataframes regardless of theme */
    [data-testid="stDataFrame"] { border: 1.5px solid var(--ink); border-radius: 2px; }
    [data-testid="stDataFrame"] * { color: var(--ink) !important; }
    [data-testid="stDataFrame"] [data-testid="stHeader"] { background: var(--paper) !important; }

    [data-testid="stAlert"] { border-radius: 2px; border: 1.5px solid var(--ink); border-left: 5px solid var(--ink); background: var(--paper-raised) !important; }
    [data-testid="stAlert"] p { color: var(--ink) !important; }

    code { font-family: 'IBM Plex Mono', monospace !important; background: var(--paper) !important; border: 1px solid var(--line); border-radius: 2px; padding: 0.1rem 0.45rem !important; color: var(--rust) !important; font-size: 0.85em; }

    [data-testid="stTabs"] [data-baseweb="tab-list"] { border-bottom: 1.5px solid var(--line); gap: 1.5rem; }
    [data-testid="stTabs"] [data-baseweb="tab"] { font-family: 'Inter', sans-serif; font-weight: 600; color: var(--ink-soft); background: transparent; }
    [data-testid="stTabs"] [data-baseweb="tab"] p { color: inherit !important; }
    [data-testid="stTabs"] [aria-selected="true"] { color: var(--rust) !important; border-bottom: 2.5px solid var(--rust) !important; }

    [data-testid="stProgress"] > div > div { background: var(--rust) !important; }
    [data-testid="stProgress"] { border: 1px solid var(--ink); border-radius: 2px; overflow: hidden; }

    [data-testid="stCaptionContainer"] { color: var(--ink-soft) !important; font-size: 0.88rem; }
    [data-testid="stCaptionContainer"] p { color: var(--ink-soft) !important; }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header[data-testid="stHeader"] { background: var(--paper) !important; }

    hr { border: none !important; border-top: 1.5px dashed var(--line) !important; margin: 2rem 0 !important; }

    [data-testid="stFileUploaderDropzone"] { background: var(--paper-raised) !important; border: 1.5px dashed var(--ink-soft) !important; border-radius: 3px !important; }
    [data-testid="stFileUploaderDropzone"] * { color: var(--ink) !important; }

    .stSelectbox [data-baseweb="select"] > div { border: 1.5px solid var(--ink) !important; border-radius: 2px !important; background: var(--paper-raised) !important; }
    .stSelectbox [data-baseweb="select"] * { color: var(--ink) !important; }

    [data-testid="stExpander"] { border: 1.5px solid var(--ink) !important; border-radius: 2px !important; background: var(--paper-raised); }
    [data-testid="stExpander"] summary { font-family: 'Inter', sans-serif; font-weight: 500; color: var(--ink) !important; }

    .sage-eyebrow { font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.15em; color: var(--rust); font-weight: 600; margin-bottom: 0.4rem; display: block; }

    /* model badge chips on Model Arena page */
    .sage-model-chip {
        background: var(--paper-raised); border: 1.5px solid var(--ink); border-radius: 2px;
        padding: 0.6rem; text-align: center; font-family: 'Inter', sans-serif; font-weight: 600;
        font-size: 0.85rem; color: var(--ink) !important; box-shadow: 2px 2px 0px var(--ink);
    }
    </style>
    """, unsafe_allow_html=True)