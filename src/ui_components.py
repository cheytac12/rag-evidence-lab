"""Streamlit design system and seven-page RAG Evidence Lab interface."""
from __future__ import annotations

from dataclasses import asdict, replace
import csv
import hashlib
import html
import io
import json
from pathlib import Path
import re
import statistics
from typing import Any

from .answer_metrics import claim_metric_summary
from .chunking import CHUNK_OVERLAPS, chunk_pages
from .document_loader import ingest_pdfs
from .embeddings import DEFAULT_EMBEDDING, retrieve_dense
from .experiment_runner import (aggregate_question_results, benchmark_counts,
                                file_sha256, load_benchmark, load_frozen_configuration,
                                require_formal_benchmark)
from .generator import (MAX_OUTPUT_TOKENS, PROMPT_VERSION, TRANSMISSION_NOTICE, configured_model,
                        configured_provider, generate_answer, llm_configured,
                        missing_configuration_message)
from .retrieval_bm25 import retrieve_bm25
from .retrieval_hybrid import retrieve_hybrid
from .reranker import RERANKER_MODEL, rerank_passages
from .schemas import (BenchmarkQuestion, ChunkRecord, PageRecord, RetrievedPassage,
                      read_jsonl, validate_chunk_record, validate_page_record,
                      validate_question_record, write_jsonl)
from .storage import ExperimentStore


NAV = ["Overview", "Corpus", "Ask", "Experiments", "Claims", "Retrieval debugger", "Exports"]
PAGE_INFO = {
    "Overview": ("Research overview", "A controlled workspace for document-grounded answers and retrieval evaluation."),
    "Corpus": ("Corpus", "Add and inspect any PDF collection; extraction and indexing run locally."),
    "Ask": ("Ask your documents", "Retrieve traceable passages and generate a cited answer from your selected collection."),
    "Experiments": ("Experiment results", "Compare retrieval and answer metrics for the selected data and benchmark."),
    "Claims": ("Claim-level evidence review", "Inspect automated claim labels, citations, and source passages."),
    "Retrieval debugger": ("Retrieval debugger", "Compare retrieved passages with annotated source spans."),
    "Exports": ("Exports", "Download filtered results, metrics, run metadata, and validation records."),
}


def _style(st):
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600;700&family=Geist+Mono:wght@400;500;600&family=Instrument+Serif:ital@1&display=swap');
    :root {
      color-scheme:dark;
      --bg:#07070B; --surface:#0D0D14; --surface-2:#14141D; --surface-3:#1B1B27;
      --text:#F5F5F7; --secondary:#B9B9CA; --muted:#9A9AB0; --subtle:#77778D;
      --border:rgba(255,255,255,.08); --border-strong:rgba(255,255,255,.16);
      --primary:#7C5CFF; --pink:#FF5CAA; --amber:#FFB45C;
      --success:#34D399; --info:#22D3EE; --warning:#FBBF24; --error:#F87171;
      --code-bg:#101018;
    }
    html, body { background:var(--bg); color:var(--text); font-family:"Geist",Inter,system-ui,-apple-system,"Segoe UI",sans-serif; }
    .stApp, [data-testid="stAppViewContainer"] {
      background:
        radial-gradient(ellipse 48rem 32rem at 8% -8%, rgba(124,92,255,.17), transparent 72%),
        radial-gradient(ellipse 42rem 26rem at 92% 2%, rgba(255,92,170,.10), transparent 72%),
        radial-gradient(ellipse 32rem 24rem at 58% 62%, rgba(255,180,92,.045), transparent 75%),
        linear-gradient(180deg,#08080D 0%,var(--bg) 44%,#090910 100%) !important;
      background-size:120% 120%,120% 120%,100% 100%,100% 100%;
      background-position:0% 0%,100% 0%,50% 60%,center;
      animation:aurora-drift 28s ease-in-out infinite alternate;
      color:var(--text) !important; font-family:"Geist",Inter,system-ui,-apple-system,"Segoe UI",sans-serif;
    }
    .stApp::before {
      content:""; position:fixed; inset:0; z-index:0; pointer-events:none; opacity:.035;
      background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 180 180'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.86' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='.55'/%3E%3C/svg%3E");
    }
    .stApp::after {
      content:""; position:fixed; inset:0; z-index:0; pointer-events:none; opacity:.28;
      background-image:radial-gradient(rgba(255,255,255,.16) .65px,transparent .65px);
      background-size:28px 28px; mask-image:linear-gradient(to bottom,black,transparent 78%);
    }
    [data-testid="stAppViewContainer"] > .main { position:relative; z-index:1; background:transparent !important; }
    /* The compact glass header keeps Streamlit's sidebar toggle available. */
    header[data-testid="stHeader"] {
      display:flex !important; position:sticky !important; top:0; width:100% !important;
      height:48px !important; min-height:48px !important; visibility:visible !important;
      background:rgba(7,7,11,.72) !important; backdrop-filter:blur(20px);
      box-shadow:0 1px 0 var(--border) !important; border:0 !important; z-index:20;
    }
    header[data-testid="stHeader"] [data-testid="stToolbar"] { background:transparent !important; min-height:48px !important; }
    header[data-testid="stHeader"] [data-testid="stAppDeployButton"],
    header[data-testid="stHeader"] [data-testid="stMainMenu"] { display:none !important; }
    header[data-testid="stHeader"] [data-testid="stStatusWidget"] { color:var(--secondary) !important; background:transparent !important; }
    header[data-testid="stHeader"] [data-testid="stStatusWidget"] button { color:var(--text) !important; background:var(--surface-2) !important; border:1px solid var(--border-strong) !important; border-radius:10px; }
    .block-container { max-width:1280px; padding-top:2rem !important; padding-bottom:4rem; color:var(--text); }
    h1,h2,h3,h4,h5,h6 { color:var(--text) !important; font-family:"Geist",Inter,system-ui,sans-serif; }
    h1 { font-size:clamp(2.25rem,4vw,3.35rem) !important; line-height:1.03 !important; letter-spacing:-.045em; font-weight:600 !important; }
    h2 { font-size:21px !important; line-height:1.3 !important; letter-spacing:-.025em; font-weight:600 !important; margin-top:2rem !important; }
    h3 { font-size:16px !important; line-height:1.4 !important; letter-spacing:-.01em; font-weight:600 !important; }
    p, li, .stMarkdown, [data-testid="stMarkdownContainer"] { color:var(--text); font-size:14px; line-height:1.65; }
    a { color:#B9A9FF !important; text-decoration-thickness:1px; text-underline-offset:3px; }
    a:hover { color:#E0D8FF !important; }
    [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * { color:var(--muted) !important; opacity:1 !important; font-size:12px; line-height:1.55; }
    [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] * { color:var(--secondary) !important; opacity:1 !important; font-weight:500; font-size:13px; }
    [data-testid="stMetric"] {
      background:linear-gradient(145deg,rgba(255,255,255,.045),rgba(255,255,255,.018)) !important;
      color:var(--text); border:1px solid var(--border) !important; border-radius:20px; padding:20px;
      box-shadow:inset 0 1px 0 rgba(255,255,255,.06),0 24px 60px -32px rgba(0,0,0,.8);
      transition:transform .22s cubic-bezier(.22,1,.36,1),border-color .22s,box-shadow .22s;
    }
    [data-testid="stMetric"]:hover { transform:translateY(-2px); border-color:rgba(124,92,255,.42) !important; box-shadow:inset 0 1px 0 rgba(255,255,255,.08),0 28px 64px -30px rgba(124,92,255,.18); }
    [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p { color:var(--muted) !important; opacity:1 !important; font-size:12px !important; font-weight:500 !important; line-height:1.45 !important; }
    [data-testid="stMetricValue"], [data-testid="stMetricValue"] p { color:var(--text) !important; opacity:1 !important; font-size:30px !important; font-weight:600 !important; line-height:1.2 !important; letter-spacing:-.04em; }
    div[data-testid="stVerticalBlockBorderWrapper"] {
      border-radius:20px; border-color:var(--border) !important;
      background:linear-gradient(145deg,rgba(255,255,255,.035),rgba(255,255,255,.012)) !important;
      box-shadow:inset 0 1px 0 rgba(255,255,255,.055),0 24px 60px -32px rgba(0,0,0,.75);
    }
    .eyebrow { color:#BDAEFF; font-family:"Geist Mono",ui-monospace,monospace; font-size:11px; font-weight:500; letter-spacing:.09em; text-transform:uppercase; margin-bottom:12px; }
    .page-overline { display:flex; align-items:center; gap:12px; margin-bottom:4px; }
    .page-indicator { display:inline-flex; align-items:center; gap:7px; color:var(--muted); font:11px/1.4 "Geist Mono",ui-monospace,monospace; letter-spacing:.04em; text-transform:uppercase; padding:5px 10px; border:1px solid var(--border); border-radius:999px; background:rgba(255,255,255,.025); }
    .page-dot { width:6px; height:6px; border-radius:50%; background:linear-gradient(135deg,var(--primary),var(--pink)); box-shadow:0 0 10px rgba(157,131,255,.7); }
    .page-hero { position:relative; display:flex; align-items:center; justify-content:space-between; gap:28px; min-height:190px; padding:28px 32px; margin:4px 0 30px; overflow:hidden; border:1px solid var(--border); border-radius:24px; background:linear-gradient(110deg,rgba(20,20,29,.91),rgba(13,13,20,.78)); box-shadow:inset 0 1px 0 rgba(255,255,255,.07),0 28px 70px -38px rgba(0,0,0,.9); }
    .page-hero::before { content:""; position:absolute; inset:-60% auto -70% 36%; width:52%; background:radial-gradient(ellipse,rgba(124,92,255,.16),rgba(255,92,170,.07) 42%,transparent 72%); filter:blur(12px); pointer-events:none; }
    .page-hero-copy { position:relative; z-index:2; max-width:760px; }
    .page-hero h1 { margin:0 0 10px !important; font-size:clamp(2.2rem,4.2vw,3.2rem) !important; line-height:1.04 !important; }
    .page-hero-copy > p { margin:0; max-width:680px; color:var(--muted) !important; font-size:14px; line-height:1.65; }
    .page-hero-art { position:relative; z-index:1; width:210px; min-width:160px; height:140px; margin-right:12px; }
    .hero-aura { position:absolute; inset:8% 6%; border-radius:50%; background:radial-gradient(circle at 32% 32%,rgba(255,180,92,.4),transparent 31%),radial-gradient(circle at 68% 32%,rgba(255,92,170,.55),transparent 36%),radial-gradient(circle at 48% 70%,rgba(124,92,255,.66),transparent 47%); filter:blur(6px); animation:hero-breathe 9s ease-in-out infinite alternate; }
    .hero-orbit { position:absolute; top:50%; left:50%; border:1px solid rgba(255,255,255,.18); border-radius:50%; transform:translate(-50%,-50%) rotate(-18deg); }
    .hero-orbit.one { width:205px; height:90px; }
    .hero-orbit.two { width:156px; height:126px; transform:translate(-50%,-50%) rotate(34deg); border-color:rgba(189,174,255,.24); }
    .hero-core { position:absolute; top:50%; left:50%; width:52px; height:52px; display:grid; place-items:center; transform:translate(-50%,-50%); border:1px solid rgba(255,255,255,.34); border-radius:18px; background:linear-gradient(140deg,rgba(124,92,255,.8),rgba(255,92,170,.48)); box-shadow:0 0 36px rgba(124,92,255,.32),inset 0 1px 0 rgba(255,255,255,.38); color:white; font-size:21px; font-weight:700; }
    .hero-spark { position:absolute; width:7px; height:7px; border-radius:50%; background:#F5E7FF; box-shadow:0 0 14px rgba(255,255,255,.9); }
    .hero-spark.a { top:22%; left:17%; }.hero-spark.b { right:12%; top:42%; width:5px; height:5px; background:#FFB45C; }.hero-spark.c { bottom:9%; left:54%; width:4px; height:4px; }
    @keyframes hero-breathe { from { transform:scale(.94) rotate(-5deg); opacity:.72; } to { transform:scale(1.08) rotate(7deg); opacity:1; } }
    @keyframes aurora-drift { from { background-position:0% 0%,100% 0%,50% 60%,center; } to { background-position:14% 12%,84% 16%,46% 50%,center; } }
    .subtle { color:var(--muted) !important; }
    .pill { display:inline-block; font-family:"Geist Mono",ui-monospace,monospace; font-size:11px; letter-spacing:.04em; padding:4px 10px; border-radius:999px; border:1px solid var(--border-strong); background:rgba(255,255,255,.045); color:var(--secondary); }
    .source-text { color:var(--text); white-space:pre-wrap; word-break:break-word; font-size:14px; line-height:1.75; }
    .brand-lockup { display:flex; align-items:center; gap:11px; padding:9px 4px 18px; border-bottom:1px solid var(--border); margin-bottom:14px; }
    .brand-mark { width:36px; height:36px; display:grid; place-items:center; flex:0 0 36px; border-radius:12px; color:white; font-size:19px; font-weight:700; background:linear-gradient(135deg,var(--primary),var(--pink) 66%,var(--amber)); box-shadow:0 6px 24px rgba(124,92,255,.28),inset 0 1px 0 rgba(255,255,255,.35); }
    .brand-name { color:var(--text); font-weight:600; font-size:14px; letter-spacing:-.025em; }
    .brand-kicker { display:block; color:var(--muted); font:10px/1.5 "Geist Mono",ui-monospace,monospace; letter-spacing:.08em; text-transform:uppercase; margin-top:3px; }
    .status-line { border:1px solid var(--border); background:rgba(255,255,255,.025); border-radius:12px; padding:10px 12px; margin:4px 0; }
    .status-line strong { color:var(--text); }
    .claim-card, .gold-evidence {
      border:1px solid var(--border); border-radius:16px; padding:16px 18px; margin:12px 0;
      background:linear-gradient(145deg,rgba(255,255,255,.045),rgba(255,255,255,.018));
      box-shadow:inset 0 1px 0 rgba(255,255,255,.05),0 18px 45px -30px rgba(0,0,0,.85);
    }
    .gold-evidence { border-left:3px solid var(--info); background:rgba(34,211,238,.055); }

    /* Glass sidebar with full-width keyboard-accessible navigation rows. */
    section[data-testid="stSidebar"] { background:rgba(13,13,20,.88) !important; backdrop-filter:blur(24px); color:var(--text) !important; border-right:1px solid var(--border); }
    section[data-testid="stSidebar"] > div { background:transparent !important; }
    section[data-testid="stSidebar"] h3 { color:var(--text) !important; font-size:18px !important; font-weight:600 !important; }
    section[data-testid="stSidebar"] [data-testid="stRadioGroup"] { width:100%; gap:5px; }
    section[data-testid="stSidebar"] label[data-testid="stRadioOption"] {
      display:flex !important; align-items:center; box-sizing:border-box; width:100% !important;
      min-height:42px; margin:0; padding:0 !important; border:1px solid transparent;
      border-radius:12px; background:transparent; color:var(--secondary) !important; cursor:pointer;
      transition:background .18s,border-color .18s,color .18s;
    }
    section[data-testid="stSidebar"] label[data-testid="stRadioOption"] > div {
      display:flex; align-items:center; box-sizing:border-box; width:100%; min-height:40px; padding:0 12px;
    }
    section[data-testid="stSidebar"] label[data-testid="stRadioOption"] > div > div:first-child { display:none !important; }
    section[data-testid="stSidebar"] label[data-testid="stRadioOption"] [data-testid="stMarkdownContainer"],
    section[data-testid="stSidebar"] label[data-testid="stRadioOption"] p {
      margin:0; color:var(--secondary) !important; font-size:13px !important; font-weight:500 !important; line-height:20px !important;
    }
    section[data-testid="stSidebar"] label[data-testid="stRadioOption"][data-selected="true"] {
      border-color:rgba(124,92,255,.22); background:linear-gradient(90deg,rgba(124,92,255,.15),rgba(255,92,170,.055)) !important;
      box-shadow:inset 2px 0 0 #9D83FF,inset 0 1px 0 rgba(255,255,255,.04);
    }
    section[data-testid="stSidebar"] label[data-testid="stRadioOption"][data-selected="true"] p { color:#E0D8FF !important; font-weight:600 !important; }
    section[data-testid="stSidebar"] label[data-testid="stRadioOption"]:hover { background:rgba(255,255,255,.045) !important; border-color:var(--border); }
    section[data-testid="stSidebar"] label[data-testid="stRadioOption"]:focus-within { outline:3px solid var(--primary) !important; outline-offset:2px; }
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color:var(--muted) !important; }
    section[data-testid="stSidebar"] hr { border-color:var(--border) !important; }

    /* Aurora primary actions, glass secondary actions, clear focus and disabled states. */
    .stButton > button, .stDownloadButton > button, [data-testid="stFileUploader"] button {
      min-height:42px; border-radius:12px; border:1px solid var(--border-strong) !important;
      background:rgba(255,255,255,.045) !important; color:var(--text) !important; font:inherit; font-weight:600;
      transition:transform .18s cubic-bezier(.22,1,.36,1),box-shadow .18s,border-color .18s,background .18s;
    }
    .stButton > button:hover:not(:disabled), .stDownloadButton > button:hover:not(:disabled),
    [data-testid="stFileUploader"] button:hover:not(:disabled) { background:rgba(255,255,255,.09) !important; border-color:rgba(255,255,255,.28) !important; transform:translateY(-1px); }
    .stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {
      background:linear-gradient(112deg,#7C5CFF 0%,#C64ED2 55%,#FF5CAA 100%) !important;
      border:1px solid rgba(255,255,255,.2) !important; color:#FFFFFF !important;
      box-shadow:0 8px 28px rgba(124,92,255,.24),inset 0 1px 0 rgba(255,255,255,.26);
    }
    .stButton > button[kind="primary"]:hover:not(:disabled), .stDownloadButton > button[kind="primary"]:hover:not(:disabled) {
      background:linear-gradient(112deg,#9278FF 0%,#D75FD9 55%,#FF76B7 100%) !important;
      transform:translateY(-1px); box-shadow:0 12px 34px rgba(124,92,255,.34),inset 0 1px 0 rgba(255,255,255,.3);
    }
    .stButton > button:focus-visible, .stDownloadButton > button:focus-visible,
    [data-testid="stFileUploader"] button:focus-visible,
    input:focus-visible, textarea:focus-visible, select:focus-visible, [role="combobox"]:focus-visible {
      outline:3px solid #BDAEFF !important; outline-offset:2px !important;
    }
    [data-testid="stExpandSidebarButton"] { color:var(--text) !important; background:transparent !important; }
    [data-testid="stExpandSidebarButton"] span,
    [data-testid="stExpandSidebarButton"] [data-testid="stIconMaterial"] { color:var(--text) !important; opacity:1 !important; }
    .stButton > button:disabled, .stDownloadButton > button:disabled, .stButton > button[aria-disabled="true"], .stDownloadButton > button[aria-disabled="true"] {
      background:#22222D !important; color:#A0A0B2 !important; border-color:rgba(255,255,255,.1) !important; opacity:1 !important; cursor:not-allowed !important; box-shadow:none !important;
    }
    input, textarea, select, [role="combobox"] { color:var(--text) !important; background:var(--surface-2) !important; }
    input::placeholder, textarea::placeholder { color:var(--subtle) !important; opacity:1 !important; }
    [data-baseweb="input"] > div, [data-baseweb="textarea"] > div, [data-baseweb="select"] > div {
      background:rgba(20,20,29,.92) !important; border-color:var(--border-strong) !important; color:var(--text) !important;
      border-radius:12px !important; box-shadow:inset 0 1px 0 rgba(255,255,255,.025);
    }
    [data-testid="stSelectbox"] button { color:var(--text) !important; background:transparent !important; }
    [data-baseweb="popover"], [data-baseweb="menu"], [role="listbox"], [role="option"] { color:var(--text) !important; background:var(--surface-2) !important; }
    [role="option"][aria-selected="true"] { background:rgba(124,92,255,.18) !important; color:#E0D8FF !important; }
    [data-testid="stFileUploaderDropzone"] { background:rgba(20,20,29,.55) !important; border:1px dashed rgba(255,255,255,.2) !important; border-radius:16px !important; }
    [data-testid="stFileUploaderDropzone"]:hover { border-color:rgba(157,131,255,.65) !important; background:rgba(124,92,255,.06) !important; }
    [data-testid="stDataFrame"] { overflow-x:auto; color:var(--text); border:1px solid var(--border); border-radius:14px; }
    [data-testid="stDataFrame"] iframe { color-scheme:dark; }
    table th, table td { color:var(--text) !important; }
    [data-testid="stExpander"] { border:1px solid var(--border) !important; border-radius:14px !important; background:rgba(255,255,255,.025) !important; }
    [data-testid="stExpander"] summary { color:var(--text) !important; }
    [data-testid="stDialog"] > div, [role="dialog"] { background:var(--surface) !important; border:1px solid var(--border-strong) !important; border-radius:22px !important; }
    [data-testid="stProgress"] > div > div { background:linear-gradient(90deg,var(--primary),var(--pink),var(--amber)) !important; }

    /* Dark semantic callouts keep type meaning without pastel panels. */
    [data-testid="stAlertContainer"] { border-radius:14px; color:var(--text); background:rgba(255,255,255,.025) !important; border:1px solid var(--border) !important; }
    [data-testid="stAlertContainer"]:has([data-testid="stAlertContentInfo"]) { background:rgba(34,211,238,.055) !important; border-color:rgba(34,211,238,.24) !important; }
    [data-testid="stAlertContentInfo"], [data-testid="stAlertContentInfo"] * { color:#A8F1FB !important; opacity:1 !important; }
    [data-testid="stAlertContainer"]:has([data-testid="stAlertContentSuccess"]) { background:rgba(52,211,153,.06) !important; border-color:rgba(52,211,153,.24) !important; }
    [data-testid="stAlertContentSuccess"], [data-testid="stAlertContentSuccess"] * { color:#8BE8C2 !important; opacity:1 !important; }
    [data-testid="stAlertContainer"]:has([data-testid="stAlertContentWarning"]) { background:rgba(251,191,36,.065) !important; border-color:rgba(251,191,36,.25) !important; }
    [data-testid="stAlertContentWarning"], [data-testid="stAlertContentWarning"] * { color:#FFDB80 !important; opacity:1 !important; }
    [data-testid="stAlertContainer"]:has([data-testid="stAlertContentError"]) { background:rgba(248,113,113,.065) !important; border-color:rgba(248,113,113,.25) !important; }
    [data-testid="stAlertContentError"], [data-testid="stAlertContentError"] * { color:#FFAAAA !important; opacity:1 !important; }
    code { background:var(--code-bg) !important; color:#D6CCFF !important; border:1px solid var(--border); border-radius:7px; padding:.08em .36em; }
    pre, pre code { background:var(--code-bg) !important; color:#E6E3F0 !important; }
    [data-testid="stMarkdownContainer"] hr { border-color:var(--border) !important; }
    div[style*="background:white"], div[style*="background: white"] { background:var(--surface-2) !important; color:var(--text) !important; border-color:var(--border) !important; }
    div[style*="background:#EFF6FF"] { background:rgba(34,211,238,.06) !important; border-color:var(--info) !important; color:#D9F9FC !important; }
    .stMarkdown h4 { color:var(--secondary) !important; font:500 11px/1.5 "Geist Mono",ui-monospace,monospace !important; letter-spacing:.06em; text-transform:uppercase; }
    @media(max-width:760px) {
      .block-container { padding:2.6rem 1rem 2.5rem !important; }
      [data-testid="stMetric"] { padding:14px; border-radius:16px; }
      h1 { font-size:2.25rem !important; }
      h2 { margin-top:1.5rem !important; }
      .page-hero { min-height:unset; padding:24px 20px; border-radius:20px; }
      .page-hero-art { width:116px; min-width:116px; height:112px; margin:0 -14px 0 -24px; transform:scale(.78); transform-origin:center; }
      .page-hero-copy { min-width:0; }
      .page-hero h1 { font-size:2rem !important; }
    }
    @media(max-width:480px) { .page-hero-art { display:none; } .page-hero { padding:22px 18px; } }
    @media(prefers-reduced-motion:reduce) { *,*::before,*::after { animation-duration:.01ms !important; animation-iteration-count:1 !important; scroll-behavior:auto !important; transition-duration:.01ms !important; } }
    </style>
    """, unsafe_allow_html=True)


def _safe_read(path: str | Path, validator) -> tuple[list[Any], str | None]:
    try:
        return read_jsonl(path, validator), None
    except Exception as exc:
        return [], str(exc)


def _safe_error(value: str) -> str:
    import os
    for secret_name in ("OPENAI_API_KEY", "GEMINI_API_KEY"):
        secret = os.environ.get(secret_name)
        if secret:
            value = value.replace(secret, "[REDACTED]")
    return value


def _active_data(st, label: str) -> tuple[list[PageRecord], list[ChunkRecord], list[BenchmarkQuestion], str | None]:
    if label == "DEMO":
        pages_path, chunks_path, bench_path = "data/demo/demo_pages.jsonl", "data/demo/demo_chunks.jsonl", "data/demo/demo_questions.jsonl"
    else:
        pages_path, chunks_path, bench_path = "data/processed/pages.jsonl", "data/processed/chunks.jsonl", "data/benchmark_dev.jsonl"
    pages, error = _safe_read(pages_path, validate_page_record)
    chunks, chunk_error = _safe_read(chunks_path, validate_chunk_record)
    questions, q_error = _safe_read(bench_path, validate_question_record)
    selected_demo = label == "DEMO"
    pages = [page for page in pages if page.is_demo is selected_demo]
    chunks = [chunk for chunk in chunks if chunk.is_demo is selected_demo]
    questions = [question for question in questions if question.is_demo is selected_demo]
    if label == "DEMO" and pages and not chunks:
        try:
            chunks = chunk_pages(pages, 256, 48)
        except Exception as exc:
            chunk_error = str(exc)
    return pages, chunks, questions, error or chunk_error or q_error


def _document_rows(pages: list[PageRecord]) -> list[dict[str, Any]]:
    grouped: dict[str, list[PageRecord]] = {}
    for page in pages:
        grouped.setdefault(page.doc_id, []).append(page)
    rows = []
    for doc_id, records in grouped.items():
        records.sort(key=lambda page: page.page_number)
        rows.append({"Title": records[0].title, "Filename": records[0].source_file,
                     "Pages": len(records), "Extraction": "Readable text" if any(p.text for p in records) else "No readable text",
                     "SHA-256 prefix": records[0].source_sha256[:12], "Label": "DEMO" if records[0].is_demo else "REAL",
                     "doc_id": doc_id})
    return rows


def _benchmark_readiness(path: str, pages: list[PageRecord], split: str, demo: bool = False) -> tuple[str, str]:
    try:
        records, report = load_benchmark(path, pages)
        report["split"] = split
        if not report["valid"]:
            return "Blocked", "; ".join(report["issues"])
        if demo:
            return ("Ready", f"{len(records)} isolated synthetic demo questions") if records and all(q.is_demo for q in records) else ("Needs setup", "Demo question file is empty or mixed with real records.")
        require_formal_benchmark(records, split)
        return "Ready", f"{len(records)} verified real questions"
    except Exception as exc:
        return "Needs setup", str(exc)


def _render_header(st, page: str):
    title, subtitle = PAGE_INFO[page]
    st.markdown(
        f'<section class="page-hero"><div class="page-hero-copy">'
        f'<div class="page-overline"><span class="eyebrow">RAG EVIDENCE LAB</span>'
        f'<span class="page-indicator"><span class="page-dot"></span>{html.escape(page)}</span></div>'
        f'<h1>{html.escape(title)}</h1><p>{html.escape(subtitle)}</p></div>'
        '<div class="page-hero-art" aria-hidden="true"><div class="hero-aura"></div>'
        '<div class="hero-orbit one"></div><div class="hero-orbit two"></div>'
        '<div class="hero-core">R</div><span class="hero-spark a"></span>'
        '<span class="hero-spark b"></span><span class="hero-spark c"></span></div></section>',
        unsafe_allow_html=True,
    )


def _overview(st, pages, chunks, questions, label, store):
    rows = _document_rows(pages)
    active_demo = label == "DEMO"
    runs = store.runs(data_label="demo" if active_demo else "real")
    completed = [row for row in runs if row["status"] == "complete"]
    vq = sum(q.annotation_verified for q in questions)
    cards = st.columns(4)
    for col, title, value, caption in zip(cards,
        ["Indexed documents", "Indexed chunks", "Verified benchmark questions", "Completed experiment runs"],
        [len(rows), len(chunks), vq, len(completed)],
        [f"{label} collection", f"{label} collection", f"{label} benchmark", f"{label} runs"]):
        with col:
            st.metric(title, value)
            st.caption(caption)
    st.subheader("Readiness checklist")
    dev_path = "data/demo/demo_questions.jsonl" if active_demo else "data/benchmark_dev.jsonl"
    test_path = "data/demo/demo_questions.jsonl" if active_demo else "data/benchmark_test.jsonl"
    dev_status, dev_reason = _benchmark_readiness(dev_path, pages, "dev", active_demo)
    test_status, test_reason = _benchmark_readiness(test_path, pages, "test", active_demo)
    frozen = False
    frozen_reason = "Run development experiments, then freeze the selected test configuration."
    try:
        load_frozen_configuration("results/frozen_test_config.yaml", "experiments/experiment_matrix.yaml")
        frozen = True
        frozen_reason = "Frozen snapshot checksum and source matrix are valid."
    except Exception as exc:
        frozen_reason = str(exc)
    checks = [
        ("Corpus indexed", "Ready" if rows else "Needs setup", f"{len(rows)} {label.lower()} document(s) indexed." if rows else "Add PDFs or select the isolated demo collection."),
        ("Development benchmark valid", dev_status, dev_reason),
        ("Test benchmark valid", test_status, test_reason),
        ("Model API configured", "Ready" if llm_configured() else "Needs setup", f"{configured_provider().title()} · {configured_model()} (key hidden)" if llm_configured() else missing_configuration_message()),
        ("Test configuration frozen", "Ready" if frozen else "Needs setup", frozen_reason),
    ]
    for name, status, reason in checks:
        col1, col2, col3 = st.columns([2, 1, 7])
        col1.write(name)
        col2.write(f"**{status}**")
        col3.caption(reason)
    st.subheader("Latest results")
    latest = [row for row in completed[:8]]
    if latest:
        output = []
        result_index = {item["run_id"]: item for item in store.question_results()}
        for run in latest:
            qrows = store.question_results(run["run_id"])
            recalls = [q["result"].get("retrieval_metrics_at_5", q["result"].get("retrieval_metrics", {})).get("recall") for q in qrows]
            faith = [q["result"].get("faithfulness") for q in qrows if q["result"].get("faithfulness") is not None]
            latency = [q["result"].get("latency", {}).get("total_seconds") for q in qrows if q["result"].get("latency", {}).get("total_seconds") is not None]
            settings = json.loads(run["settings_json"])
            output.append({"Run date": run["created_at"][:10], "Configuration": run["configuration_id"], "Split": run["split_name"],
                           "Question count": len(qrows), "Faithfulness": statistics.mean(faith) if faith else None,
                           "Recall@5": statistics.mean([v for v in recalls if v is not None]) if any(v is not None for v in recalls) else None,
                           "Latency (s)": statistics.median(latency) if latency else None})
        st.dataframe(output, use_container_width=True, hide_index=True)
    else:
        st.info("No completed runs yet.")
    st.subheader("Start here")
    st.markdown("1. **Add PDFs** — use the Corpus page.\n2. **Validate benchmark** — `python -m src.cli validate-benchmark --split dev`\n3. **Run development experiments** — `python -m src.cli run-experiments --matrix experiments/experiment_matrix.yaml --split dev`")


def _corpus_page(st, pages, chunks, label):
    if label == "DEMO":
        st.info("This collection contains generic synthetic records labeled DEMO. Demo records are isolated from real results and conclusions.")
    st.info("Uploaded files are processed locally and are not sent to the model provider during ingestion.")
    st.caption("PDF files only · maximum 50 MB per file")
    files = st.file_uploader("Add PDF files", type=["pdf"], accept_multiple_files=True, key="pdf_uploader")
    too_large = [file.name for file in (files or []) if file.size > 50 * 1024 * 1024]
    if too_large:
        st.error("Each PDF must be 50 MB or smaller: " + ", ".join(too_large))
    elif label == "DEMO":
        st.caption("PDF processing is disabled while the DEMO collection is selected. Switch to REAL to add your own PDFs.")
    elif not files:
        st.caption("Select one or more PDF files to enable local processing.")
    if st.button("Process PDFs", type="primary", disabled=not files or bool(too_large) or label == "DEMO"):
        target_dir = Path("data/pdfs")
        target_dir.mkdir(parents=True, exist_ok=True)
        conflict = None
        for uploaded in files:
            name = Path(uploaded.name).name
            destination = target_dir / name
            if destination.exists():
                conflict = f"A file named {name} already exists in data/pdfs. Rename the new file to preserve both originals."
                break
        if conflict:
            st.error(conflict)
        else:
            for uploaded in files:
                name = Path(uploaded.name).name
                (target_dir / name).write_bytes(uploaded.getvalue())
            with st.spinner("Extracting pages locally…"):
                try:
                    report = ingest_pdfs(target_dir, "data/processed/pages.jsonl")
                    st.session_state["corpus_message"] = f"Processed {len(report.processed_files)} PDF(s), {report.page_count} page(s); {len(report.failed_files)} failed, {len(report.empty_text_pages)} page(s) had no readable text, {len(report.duplicate_files)} duplicate(s)."
                    st.session_state["corpus_report"] = report.__dict__
                    try:
                        indexed_pages = read_jsonl("data/processed/pages.jsonl", validate_page_record)
                        indexed_chunks = chunk_pages(indexed_pages, 256, 48)
                        write_jsonl("data/processed/chunks.jsonl", indexed_chunks)
                        st.session_state["corpus_message"] += f" Created {len(indexed_chunks)} chunks at 256/48."
                    except Exception as chunk_exc:
                        st.session_state["corpus_message"] += f" Text extraction completed; chunk indexing needs setup: {chunk_exc}"
                    if report.failed_files:
                        st.error("One or more PDFs could not be processed.")
                    else:
                        st.success(st.session_state["corpus_message"])
                except Exception as exc:
                    st.error(f"PDF processing failed: {exc}")
    if st.session_state.get("corpus_message"):
        st.success(st.session_state["corpus_message"])
    report_data = st.session_state.get("corpus_report")
    if report_data:
        for failed in report_data.get("failed_files", []):
            st.error(f"Failed PDF: {failed['file']} — {failed['reason']}")
        for item in report_data.get("image_only_pages", []):
            st.warning(f"Text extraction found no readable text. OCR is not included. File: {item['file']}, page {item['page']}.")
        for item in report_data.get("empty_text_pages", []):
            if item not in report_data.get("image_only_pages", []):
                st.caption(f"No extractable text: {item['file']}, page {item['page']}.")
        for item in report_data.get("duplicate_files", []):
            st.caption(f"Duplicate PDF skipped by checksum: {item['file']} duplicates {item['duplicate_of']}.")
    rows = _document_rows(pages)
    st.subheader("Current documents")
    if rows:
        st.dataframe([{key: value for key, value in row.items() if key != "doc_id"} for row in rows], use_container_width=True, hide_index=True)
    else:
        st.info("No real PDF documents are indexed yet. The real corpus remains empty until PDFs are added.")
    if pages:
        st.subheader("Extracted page text")
        doc_names = {row["doc_id"]: f"{row['Title']} — {row['Filename']}" for row in rows}
        selected_doc = st.selectbox("Document", list(doc_names), format_func=lambda item: doc_names[item], key="page_doc")
        doc_pages = sorted([page for page in pages if page.doc_id == selected_doc], key=lambda p: p.page_number)
        selected_page = st.selectbox("Page number", [p.page_number for p in doc_pages], key="page_number")
        page = next(p for p in doc_pages if p.page_number == selected_page)
        if not page.text.strip():
            st.warning("Text extraction found no readable text. OCR is not included.")
        else:
            st.text_area("Extracted text", value=page.text, height=220, disabled=True)
        st.subheader("Chunk preview")
        size = st.selectbox("Chunk size (tokens)", [128, 256, 512], index=1, key="preview_size")
        overlap = CHUNK_OVERLAPS[size]
        st.caption(f"Overlap: {overlap} tokens")
        try:
            preview = chunk_pages(doc_pages, size, overlap)
            if preview:
                for idx, chunk in enumerate(preview, 1):
                    with st.expander(f"Preview chunk {idx} · {chunk.token_count} tokens · page {chunk.page_start}–{chunk.page_end}"):
                        st.write(chunk.text)
            else:
                st.info("No non-empty chunks were produced for this page selection.")
        except Exception as exc:
            st.warning(f"Chunk preview needs the configured Hugging Face tokenizer: {exc}")
    if label == "REAL" and rows:
        st.subheader("Remove an indexed record")
        selected = st.selectbox("Indexed document to remove", [row["doc_id"] for row in rows], format_func=lambda item: doc_names.get(item, item), key="delete_doc")
        @st.dialog("Confirm indexed record removal")
        def confirm_record_removal(doc_id):
            st.warning("This removes the indexed page and chunk records. The original PDF file will be kept.")
            if st.button("Confirm removal", type="primary"):
                current_pages, _ = _safe_read("data/processed/pages.jsonl", validate_page_record)
                write_jsonl("data/processed/pages.jsonl", [page for page in current_pages if page.doc_id != doc_id])
                existing_chunks, _ = _safe_read("data/processed/chunks.jsonl", validate_chunk_record)
                write_jsonl("data/processed/chunks.jsonl", [chunk for chunk in existing_chunks if chunk.doc_id != doc_id])
                st.success("Indexed page and chunk records removed. The original PDF file was preserved.")
                st.rerun()
            if st.button("Cancel"):
                st.rerun()

        if st.button("Remove indexed record"):
            confirm_record_removal(selected)


def _selected_chunks(label: str, doc_ids: set[str] | None = None) -> list[ChunkRecord]:
    if label == "DEMO":
        pages_path, chunks_path = "data/demo/demo_pages.jsonl", "data/demo/demo_chunks.jsonl"
    else:
        pages_path, chunks_path = "data/processed/pages.jsonl", "data/processed/chunks.jsonl"
    chunks, _ = _safe_read(chunks_path, validate_chunk_record)
    chunks = [chunk for chunk in chunks if chunk.is_demo == (label == "DEMO")]
    if not chunks and label == "DEMO":
        pages, _ = _safe_read(pages_path, validate_page_record)
        if pages:
            try:
                chunks = chunk_pages(pages, 256, 48)
                write_jsonl(chunks_path, chunks)
            except Exception:
                return []
    return [chunk for chunk in chunks if doc_ids is None or chunk.doc_id in doc_ids]


def _render_sources(st, passages: list[RetrievedPassage]):
    st.subheader("Retrieved sources")
    if not passages:
        st.info("No supporting passages were retrieved.")
        return
    for passage in passages:
        title = f"Rank {passage.rank} · {passage.title} · page {passage.page_start}–{passage.page_end}"
        with st.expander(title):
            st.caption(f"Passage ID: {passage.chunk_id} · retrieval score: {passage.raw_score:.5f}" + (f" · reranker score: {passage.reranker_score:.5f}" if passage.reranker_score is not None else ""))
            st.markdown(f'<div class="source-text">{html.escape(passage.text)}</div>', unsafe_allow_html=True)


def _is_document_overview_question(question: str) -> bool:
    """Recognize broad questions that should consult a document's opening pages."""
    text = " ".join(question.lower().split())
    refers_to_document = re.search(
        r"\b(?:this|the|a|an)\s+(?:(?:research|academic)\s+)?(?:paper|document|article|report|study|pdf)\b",
        text,
    )
    asks_for_overview = re.search(
        r"\b(?:overview|summari[sz](?:e|es|ed|ing|ation)?|about|purpose|motivation|contribution|problem|goal|aim|main idea)\b",
        text,
    )
    return bool(refers_to_document and asks_for_overview)


def _corpus_fingerprint(chunks: list[ChunkRecord]) -> str:
    material = "\n".join(
        f"{chunk.doc_id}:{chunk.chunk_id}:{chunk.source_sha256}"
        for chunk in sorted(chunks, key=lambda item: item.chunk_id)
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _retrieve_ask_candidates(
    question: str,
    chunks: list[ChunkRecord],
    method_label: str,
    top_k: int,
    use_reranker: bool,
) -> list[RetrievedPassage]:
    candidate_k = 20 if use_reranker else top_k
    method = method_label.lower()
    if method == "bm25":
        return retrieve_bm25(question, chunks, top_k=candidate_k)
    if method == "dense":
        return retrieve_dense(question, chunks, top_k=candidate_k, model_id=DEFAULT_EMBEDDING)
    # Fusion should see a broad enough set from both methods before truncating
    # to the user's requested result count.
    return retrieve_hybrid(
        question,
        chunks,
        top_k=candidate_k,
        model_id=DEFAULT_EMBEDDING,
        candidate_k=max(candidate_k, 20),
    )


def _add_opening_page_context(
    question: str,
    chunks: list[ChunkRecord],
    candidates: list[RetrievedPassage],
    top_k: int,
) -> tuple[list[RetrievedPassage], bool]:
    """Add lexical matches from the opening pages for a single-document overview query."""
    if not _is_document_overview_question(question) or len({chunk.doc_id for chunk in chunks}) != 1:
        return candidates, False
    opening_chunks = [chunk for chunk in chunks if chunk.page_start <= 2]
    opening_k = min(3, max(1, top_k - 1))
    opening = retrieve_bm25(question, opening_chunks, top_k=opening_k)
    if not opening:
        return candidates, False
    opening_ids = {passage.chunk_id for passage in opening}
    combined = opening + [passage for passage in candidates if passage.chunk_id not in opening_ids]
    merged = [
        replace(
            passage,
            rank=rank,
            method="dense+opening-page-context" if passage.chunk_id in opening_ids else passage.method,
        )
        for rank, passage in enumerate(combined, start=1)
    ]
    return merged, True


def _ask_page(st, pages, chunks, label):
    st.caption(f"Active data source: {label} · {len(pages)} pages · {len(chunks)} chunks")
    left, right = st.columns([7, 5], gap="large")
    with right:
        st.subheader("Retrieval settings")
        doc_groups = _document_rows(pages)
        doc_map = {row["doc_id"]: f"{row['Title']} — {row['Filename']} [{row['doc_id'][:8]}]" for row in doc_groups}
        all_option = "All indexed documents"
        collection = st.selectbox("Document collection", [all_option] + list(doc_map.values()), key="ask_collection")
        method_label = st.selectbox("Retrieval method", ["BM25", "Dense", "Hybrid"], key="ask_method")
        top_k = st.selectbox("Top-k passages", [3, 5, 10], index=1, key="ask_top_k")
        use_reranker = st.toggle("Use reranker", value=False, key="ask_reranker")
        selected_collection_doc_ids = {
            doc_id for doc_id, title in doc_map.items()
            if collection == all_option or title == collection
        }
        overview_context = (
            method_label == "Dense"
            and len(selected_collection_doc_ids) == 1
            and _is_document_overview_question(st.session_state.get("ask_question", ""))
        )
        request_context = {
            "data_source": label,
            "document_ids": sorted({page.doc_id for page in pages}),
            "question": st.session_state.get("ask_question", "").strip(),
            "collection": collection,
            "method": method_label,
            "top_k": top_k,
            "reranker": use_reranker,
            "overview_context": overview_context,
            "corpus_fingerprint": _corpus_fingerprint(chunks),
            "provider": configured_provider(),
            "model": configured_model(),
            "prompt_version": PROMPT_VERSION,
        }
        st.caption(f"Embedding model: {DEFAULT_EMBEDDING} (dense/hybrid)\n\nReranker model: {RERANKER_MODEL}")
        if overview_context:
            st.caption("For this overview question, matching passages from the document's opening pages will also be included.")
        st.subheader("Evidence")
        saved = st.session_state.get("ask_result")
        if saved and saved.get("request_context") != request_context:
            st.session_state["ask_result"] = None
            saved = None
        if saved and saved.get("passages"):
            retrieval_label = saved.get("retrieval_label", f"{method_label} retrieval")
            st.caption(f"Result uses {label} data · {retrieval_label}")
            _render_sources(st, saved["passages"])
        elif not saved:
            st.info("Retrieved passages and citations will appear here after you submit a question.")
    with left:
        st.text_area("Your question", key="ask_question", height=130, placeholder="Ask a question grounded in the selected documents.")
        st.warning(TRANSMISSION_NOTICE)
        question = st.session_state.get("ask_question", "").strip()
        if not pages:
            st.caption("Add PDFs on the Corpus page or switch to the isolated DEMO collection before retrieving.")
        elif not question:
            st.caption("Enter a question to enable evidence retrieval.")
        elif pages and not chunks:
            st.warning("This collection has extracted pages but no indexed chunks. Open Corpus and process the PDFs again before asking.")
        if st.button("Retrieve evidence and answer", type="primary", use_container_width=True, disabled=not question or not pages):
                selected_doc_ids = None
                if collection != all_option:
                    selected_doc_ids = {doc_id for doc_id, title in doc_map.items() if title == collection}
                selected = [chunk for chunk in chunks if selected_doc_ids is None or chunk.doc_id in selected_doc_ids]
                if not selected:
                    selected = _selected_chunks(label, selected_doc_ids)
                overview_context_used = False
                try:
                    with st.spinner("Retrieving evidence…"):
                        candidates = _retrieve_ask_candidates(
                            question, selected, method_label, top_k, use_reranker
                        )
                        if overview_context:
                            candidates, overview_context_used = _add_opening_page_context(
                                question, selected, candidates, top_k
                            )
                        passages = rerank_passages(question, candidates, top_k=top_k) if use_reranker else candidates[:top_k]
                    state = {"passages": passages, "answer": None, "error_state": None, "citations": [], "uncertainties": []}
                    state["request_context"] = request_context
                    if overview_context_used:
                        state["retrieval_label"] = "Dense retrieval + opening-page context"
                    if not passages:
                        state["answer"] = "The indexed documents do not provide enough evidence to answer this question."
                        state["error_state"] = "no_evidence"
                    elif not llm_configured():
                        state["error_state"] = "missing_config"
                        state["error_message"] = missing_configuration_message()
                    else:
                        try:
                            with st.spinner("Generating a cited answer…"):
                                generated = generate_answer(question, passages)
                            state.update(generated)
                        except Exception as exc:
                            state["error_state"] = "generation_failure"
                            state["error_message"] = _safe_error(f"Answer generation failed: {exc}")
                    st.session_state["ask_result"] = state
                    st.rerun()
                except Exception as exc:
                    st.session_state["ask_result"] = {"passages": [], "answer": None, "error_state": "retrieval_failure", "error_message": _safe_error(f"Retrieval failed: {exc}"), "request_context": request_context}
                    st.rerun()
        saved = st.session_state.get("ask_result")
        if saved and saved.get("request_context") != request_context:
            st.session_state["ask_result"] = None
            saved = None
        if saved:
            if saved.get("error_state") == "retrieval_failure":
                st.error(saved.get("error_message", "Retrieval failed."))
            elif saved.get("error_state") == "missing_config":
                st.info(saved.get("error_message", missing_configuration_message()))
            elif saved.get("error_state") == "generation_failure":
                st.error(saved.get("error_message", "Answer generation failed."))
            elif saved.get("error_state") == "no_evidence":
                st.warning("The indexed documents do not provide enough evidence to answer this question.")
            if saved.get("answer"):
                st.subheader("Answer")
                st.write(saved["answer"])
                for citation in saved.get("citations", []):
                    pid = citation.get("passage_id")
                    found = next((p for p in saved.get("passages", []) if p.chunk_id == pid), None)
                    if found:
                        st.caption(f"Citation for: {citation.get('claim_text','Claim')} — {found.title}, page {found.page_start}–{found.page_end} ({'valid ID' if citation.get('citation_present') else 'invalid ID'})")
                    else:
                        st.error(f"Invalid citation ID: {pid}")
                uncertainties = saved.get("uncertainties", [])
                st.caption("Uncertainty: " + ("; ".join(uncertainties) if uncertainties else "No additional uncertainty was returned."))


def _all_filtered_rows(store: ExperimentStore, label: str | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    runs = store.runs(data_label=("demo" if label == "DEMO" else "real") if label else None)
    run_map = {run["run_id"]: run for run in runs}
    rows = []
    for row in store.question_results():
        run = run_map.get(row["run_id"])
        if not run:
            continue
        item = {**row, "run": run, "settings": json.loads(run["settings_json"]), "result": row["result"]}
        rows.append(item)
    return runs, rows


def _filters(st, runs, rows, key_prefix="exp"):
    st.subheader("Filters")
    col = st.columns(4)
    split = col[0].selectbox("Split", ["All"] + sorted({r["split_name"] for r in runs}), key=f"{key_prefix}_split")
    label = col[1].selectbox("Data label", ["real", "demo"], index=1 if key_prefix and st.session_state.get("active_data_label") == "DEMO" else 0, key=f"{key_prefix}_label")
    run_id = col[2].selectbox("Run ID", ["All"] + sorted({r["run_id"] for r in runs}), key=f"{key_prefix}_run")
    config_id = col[3].selectbox("Configuration ID", ["All"] + sorted({r["configuration_id"] for r in runs}), key=f"{key_prefix}_config")
    col = st.columns(5)
    embedding = col[0].selectbox("Embedding model", ["All"] + sorted({r["settings"].get("embedding_model") or "None" for r in rows}), key=f"{key_prefix}_embedding")
    retriever = col[1].selectbox("Retriever", ["All"] + sorted({r["settings"].get("retriever", "") for r in rows}), key=f"{key_prefix}_retriever")
    size = col[2].selectbox("Chunk size", ["All"] + sorted({r["settings"].get("chunk_size") for r in rows if r["settings"].get("chunk_size")}), key=f"{key_prefix}_size")
    top_k = col[3].selectbox("Top-k", ["All"] + sorted({r["settings"].get("top_k") for r in rows if r["settings"].get("top_k")}), key=f"{key_prefix}_topk")
    rerank = col[4].selectbox("Reranking", ["All", "on", "off"], key=f"{key_prefix}_rerank")
    status = col[4].selectbox("Status", ["All"] + sorted({r["status"] for r in runs}), key=f"{key_prefix}_status")
    selected_runs = []
    for run in runs:
        settings = json.loads(run["settings_json"])
        if split != "All" and run["split_name"] != split: continue
        if run["data_label"] != label: continue
        if run_id != "All" and run["run_id"] != run_id: continue
        if config_id != "All" and run["configuration_id"] != config_id: continue
        if embedding != "All" and (settings.get("embedding_model") or "None") != embedding: continue
        if retriever != "All" and settings.get("retriever") != retriever: continue
        if size != "All" and settings.get("chunk_size") != size: continue
        if top_k != "All" and settings.get("top_k") != top_k: continue
        if rerank != "All" and bool(settings.get("reranker")) != (rerank == "on"): continue
        if status != "All" and run["status"] != status: continue
        selected_runs.append(run)
    selected_ids = {run["run_id"] for run in selected_runs}
    selected_rows = [row for row in rows if row["run_id"] in selected_ids]
    return selected_runs, selected_rows


def _group_results(runs, rows):
    grouped = []
    for run in runs:
        qrows = [r["result"] for r in rows if r["run_id"] == run["run_id"]]
        settings = json.loads(run["settings_json"])
        def mean(values):
            vals = [float(v) for v in values if v is not None]
            return statistics.mean(vals) if vals else None
        recall = mean([q.get("retrieval_metrics_at_5", q.get("retrieval_metrics", {})).get("recall") for q in qrows])
        hit = mean([q.get("retrieval_metrics", {}).get("hit") for q in qrows])
        mrr = mean([q.get("retrieval_metrics", {}).get("mrr") for q in qrows])
        faith = mean([q.get("faithfulness") for q in qrows])
        correctness_values = [q.get("correctness_automated", {}).get("score") if isinstance(q.get("correctness_automated"), dict) else q.get("correctness_automated") for q in qrows]
        correctness = mean(correctness_values)
        unsupported = mean([q.get("unsupported_answer_rate") for q in qrows])
        latency_values = [q.get("latency", {}).get("total_seconds") for q in qrows if q.get("latency", {}).get("total_seconds") is not None]
        latency = statistics.median(latency_values) if latency_values else None
        grouped.append({"configuration_id": run["configuration_id"], "run_id": run["run_id"], "split": run["split_name"], "data_label": run["data_label"],
                        "chunk_size": settings.get("chunk_size"), "embedding": settings.get("embedding_model"), "retriever": settings.get("retriever"),
                        "top_k": settings.get("top_k"), "reranker": settings.get("reranker", False), "question_count": len(qrows),
                        "Recall@5": recall, "Hit@5": hit, "MRR": mrr, "Faithfulness": faith, "Correctness": correctness,
                        "Unsupported-answer rate": unsupported, "Median latency (s)": latency, "status": run["status"],
                        "settings": settings})
    return grouped


def _chart(st, px, data, kind, title, **kwargs):
    st.markdown(f"#### {title}")
    if not data:
        st.info("No data for this chart under the current filters.")
        return
    try:
        fig = getattr(px, kind)(data, **kwargs)
        fig.update_layout(
            margin=dict(l=8, r=8, t=36, b=8), legend_title_text="Retriever",
            plot_bgcolor="#0D0D14", paper_bgcolor="#0D0D14",
            font=dict(family='Geist,Inter,system-ui,-apple-system,"Segoe UI",sans-serif', color="#B9B9CA", size=13),
            colorway=["#9B83FF", "#FF6EAF", "#34D399", "#22D3EE", "#FFB45C", "#B9B9CA"],
            hoverlabel=dict(bgcolor="#1B1B27", bordercolor="rgba(255,255,255,.16)", font=dict(color="#F5F5F7")),
        )
        fig.update_xaxes(color="#9A9AB0", title_font_color="#B9B9CA", tickfont_color="#9A9AB0",
                         gridcolor="rgba(255,255,255,.08)", zerolinecolor="rgba(255,255,255,.16)", linecolor="rgba(255,255,255,.12)", automargin=True)
        fig.update_yaxes(color="#9A9AB0", title_font_color="#B9B9CA", tickfont_color="#9A9AB0",
                         gridcolor="rgba(255,255,255,.08)", zerolinecolor="rgba(255,255,255,.16)", linecolor="rgba(255,255,255,.12)", automargin=True)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as exc:
        st.info(f"Chart unavailable for the selected data: {exc}")


def _experiments_page(st, store, label):
    runs, rows = _all_filtered_rows(store)
    chosen_runs, chosen_rows = _filters(st, runs, rows)
    groups = _group_results(chosen_runs, chosen_rows)
    active_split = ", ".join(sorted({r["split_name"] for r in chosen_runs})) or "no selected split"
    active_label = ", ".join(sorted({r["data_label"].upper() for r in chosen_runs})) or label
    st.caption(f"Active results: {active_label} · split: {active_split} · {len(chosen_rows)} question row(s) · {len(groups)} selected configuration(s)")
    st.caption("Results apply only to the selected corpus, benchmark, model, and settings. A single highest metric does not establish a general winner.")
    card = st.columns(4)
    for col, metric, title in zip(card, ["Faithfulness", "Correctness", "Recall@5", "Median latency (s)"], ["Faithfulness", "Answer correctness", "Recall@5", "Median total latency"]):
        values = [row[metric] for row in groups if row.get(metric) is not None]
        value = statistics.median(values) if metric == "Median latency (s)" and values else statistics.mean(values) if values else None
        col.metric(title, f"{value:.3f}" if value is not None else "Not available")
        col.caption(f"{len(values)} scored question(s) · {active_label} · split: {active_split}")
    st.subheader("Results")
    columns = ["configuration_id", "chunk_size", "embedding", "retriever", "top_k", "reranker", "question_count", "Recall@5", "MRR", "Faithfulness", "Correctness", "Unsupported-answer rate", "Median latency (s)", "status"]
    if groups:
        st.dataframe([{k: row.get(k) for k in columns} for row in groups], use_container_width=True, hide_index=True)
    else:
        st.info("No results match the current filters.")
    st.subheader("Metric charts")
    try:
        import pandas as pd
        import plotly.express as px
        df = pd.DataFrame(groups)
        col1, col2 = st.columns(2)
        with col1:
            data = [r for r in groups if r.get("Recall@5") is not None and r.get("chunk_size") is not None]
            _chart(st, px, data, "bar", "Recall@5 by chunk size", x="chunk_size", y="Recall@5", color="retriever", barmode="group", hover_data=["configuration_id", "question_count"])
            data = [r for r in groups if r.get("Faithfulness") is not None and r.get("Median latency (s)") is not None]
            _chart(st, px, data, "scatter", "Faithfulness versus median total latency", x="Median latency (s)", y="Faithfulness", color="retriever", hover_data=["configuration_id", "question_count"])
            data = [r for r in groups if r.get("Correctness") is not None and r.get("Faithfulness") is not None]
            _chart(st, px, data, "scatter", "Correctness versus faithfulness", x="Correctness", y="Faithfulness", color="retriever", hover_data=["configuration_id", "question_count"])
        with col2:
            data = [r for r in groups if r.get("MRR") is not None]
            _chart(st, px, data, "bar", "MRR by retriever", x="retriever", y="MRR", color="retriever", hover_data=["configuration_id", "question_count"])
            data = [r for r in groups if r.get("Unsupported-answer rate") is not None]
            _chart(st, px, data, "bar", "Unsupported-answer rate by configuration", x="configuration_id", y="Unsupported-answer rate", color="retriever", hover_data=["question_count"])
    except ImportError:
        st.info("Install pandas and Plotly to display interactive charts.")
    st.subheader("Configuration comparison")
    selectable = {row["configuration_id"]: row for row in groups}
    ids = st.multiselect("Select up to four configurations", list(selectable), max_selections=4, key="comparison_configs")
    if ids:
        cols = st.columns(len(ids))
        for col, cid in zip(cols, ids):
            with col:
                st.markdown(f"**{cid}**")
                st.json(selectable[cid]["settings"])


def _claim_page(st, store, label):
    runs = [r for r in store.runs(data_label="demo" if label == "DEMO" else "real") if r["status"] == "complete"]
    if not runs:
        st.info("No completed experiment runs are available for claim review.")
        return
    run_map = {r["run_id"]: r for r in runs}
    run_id = st.selectbox("Completed run", list(run_map), format_func=lambda rid: f"{rid} · {run_map[rid]['configuration_id']} · {run_map[rid]['split_name']}", key="claim_run")
    qrows = store.question_results(run_id)
    if not qrows:
        st.info("The selected run has no stored question results.")
        return
    result_by_q = {r["question_id"]: r["result"] for r in qrows}
    qid = st.selectbox("Question", list(result_by_q), format_func=lambda key: f"{key}: {result_by_q[key].get('question','')[:80]}", key="claim_q")
    result = result_by_q[qid]
    st.caption(f"Data label: {run_map[run_id]['data_label'].upper()} · Split: {run_map[run_id]['split_name']}")
    st.markdown(f"**Question:** {result.get('question','')}")
    st.markdown(f"**Generated answer:** {result.get('answer') or 'No generated answer stored.'}")
    if result.get("reference_answer"):
        st.markdown(f"**Reference answer:** {result['reference_answer']}")
    evaluations = result.get("claim_evaluations", [])
    claims = result.get("claims", [])
    citations = result.get("citations", [])
    if not claims:
        st.info("Not available — no factual claims were stored.")
    else:
        counts = {label_name: sum(item.get("label") == label_name for item in evaluations) for label_name in ("SUPPORTED", "UNSUPPORTED", "CONTRADICTED")}
        faithfulness = result.get("faithfulness")
        st.markdown(f"**Claim summary:** {counts['SUPPORTED']} supported · {counts['UNSUPPORTED']} unsupported · {counts['CONTRADICTED']} contradicted · faithfulness: {faithfulness if faithfulness is not None else 'Not available'}")
        filt = st.selectbox("Filter claims", ["All claims", "Supported", "Unsupported", "Contradicted", "Invalid citation"], key="claim_filter")
        eval_map = {item.get("claim_text"): item for item in evaluations}
        citation_by_claim: dict[str, list[dict[str, Any]]] = {}
        for citation in citations:
            citation_by_claim.setdefault(citation.get("claim_text", ""), []).append(citation)
        passage_by_id = {p.get("chunk_id"): p for p in result.get("retrieved_passages", [])}
        for claim in claims:
            evaluation = eval_map.get(claim.get("claim_text"), {})
            cits = citation_by_claim.get(claim.get("claim_text"), [])
            validity = claim.get("citation_id_validity", {})
            invalid = any(not c.get("citation_present") for c in cits) or any(not is_valid for is_valid in validity.values())
            label_name = evaluation.get("label", "UNEVALUATED")
            if filt == "Supported" and label_name != "SUPPORTED": continue
            if filt == "Unsupported" and label_name != "UNSUPPORTED": continue
            if filt == "Contradicted" and label_name != "CONTRADICTED": continue
            if filt == "Invalid citation" and not invalid: continue
            color = {"SUPPORTED": "#34D399", "UNSUPPORTED": "#FBBF24", "CONTRADICTED": "#F87171"}.get(label_name, "#9A9AB0")
            st.markdown(f'<div class="claim-card"><b>{html.escape(claim.get("claim_text",""))}</b><br><span style="color:{color};font-weight:700">{html.escape(label_name)}</span><br><span>{html.escape(evaluation.get("explanation", "No automated evaluation is stored."))}</span></div>', unsafe_allow_html=True)
            support_text = "supporting" if any(c.get("citation_supporting") for c in cits) else "not established as supporting" if cits else "none"
            st.caption("Cited passage IDs: " + (", ".join(claim.get("citation_ids", [])) or "none") + " · Citation validity: " + ("invalid" if invalid else "valid/present" if cits else "none") + f" · Citation support: {support_text}")
            ids = set(evaluation.get("supporting_passage_ids", []))
            for pid in sorted(ids | {c.get("passage_id") for c in cits if c.get("passage_id")}):
                passage = passage_by_id.get(pid)
                if passage:
                    with st.expander(f"Evidence · {passage.get('title')} · page {passage.get('page_start')}–{passage.get('page_end')} · {pid}"):
                        st.write(passage.get("text", ""))
    review = store.review(run_id, qid)
    if review:
        st.caption(f"Reviewed by {review['reviewer']} at {review['reviewed_at']}")
    reviewer = st.text_input("Reviewer name or label", key="reviewer_label")
    if st.button("Mark reviewed", disabled=not reviewer.strip()):
        store.mark_reviewed(run_id, qid, reviewer)
        st.success("Review saved. Automated claim labels were not changed.")


def _debugger_page(st, store, label):
    runs = [r for r in store.runs(data_label="demo" if label == "DEMO" else "real") if r["status"] in {"complete", "blocked"}]
    if not runs:
        st.info("No experiment runs are available for retrieval debugging.")
        return
    configs = sorted({r["configuration_id"] for r in runs})
    selected_config = st.selectbox("Configuration", configs, key="debug_config")
    runs = [r for r in runs if r["configuration_id"] == selected_config]
    run_map = {r["run_id"]: r for r in runs}
    run_id = st.selectbox("Run", list(run_map), key="debug_run")
    run = run_map[run_id]
    settings = json.loads(run["settings_json"])
    st.caption(f"Configuration: {run['configuration_id']} · {settings.get('retriever')} · {run['data_label'].upper()} · {run['split_name']}")
    rows = store.question_results(run_id)
    if not rows:
        st.info("No question results were recorded.")
        return
    qrow = {row["question_id"]: row["result"] for row in rows}
    qid = st.selectbox("Question", list(qrow), format_func=lambda key: f"{key}: {qrow[key].get('question','')[:80]}", key="debug_question")
    result = qrow[qid]
    st.markdown(f"**Question:** {result.get('question','')}")
    st.markdown(f"Answerability: **{'Answerable' if result.get('answerable') else 'Unanswerable'}** · Type: **{result.get('question_type','Unknown')}**")
    gold = result.get("required_evidence", [])
    if not gold:
        st.info("No evidence annotated.")
    else:
        st.subheader("Annotated required evidence")
        for span in gold:
            st.markdown(f'<div class="gold-evidence"><b>GOLD · {html.escape(span.get("doc_id", ""))} · page {span.get("page_start")}–{span.get("page_end")}</b><br>{html.escape(span.get("quote", ""))}</div>', unsafe_allow_html=True)
    passages = result.get("retrieved_passages", [])
    missed = []
    for span in gold:
        overlaps = [p for p in passages if p.get("doc_id") == span.get("doc_id") and p.get("page_start",0) <= span.get("page_end",0) and p.get("page_end",0) >= span.get("page_start",0) and p.get("char_start",0) < span.get("char_end",0) and p.get("char_end",0) > span.get("char_start",0)]
        if not overlaps: missed.append(span)
    st.subheader("Retrieved passages")
    for passage in sorted(passages, key=lambda p: p.get("rank", 0)):
        overlaps = [span for span in gold if passage.get("doc_id") == span.get("doc_id") and passage.get("page_start",0) <= span.get("page_end",0) and passage.get("page_end",0) >= span.get("page_start",0) and passage.get("char_start",0) < span.get("char_end",0) and passage.get("char_end",0) > span.get("char_start",0)]
        label_text = "Overlaps gold source span" if overlaps else "No gold source-span overlap"
        with st.expander(f"Rank {passage.get('rank')} · {passage.get('title')} · page {passage.get('page_start')}–{passage.get('page_end')} · {label_text}"):
            st.caption(f"Score: {passage.get('raw_score')} · {label_text}")
            st.write(passage.get("text", ""))
    st.subheader("Missed evidence")
    if not gold:
        st.info("No evidence annotated.")
    elif missed:
        for span in missed:
            st.warning(f"{span.get('doc_id')} · pages {span.get('page_start')}–{span.get('page_end')} · {span.get('quote')}")
    else:
        st.success("Every annotated span overlaps at least one retrieved passage.")


def _export_page(st, store, label):
    runs, rows = _all_filtered_rows(store)
    selected_runs, selected_rows = _filters(st, runs, rows, key_prefix="export")
    groups = _group_results(selected_runs, selected_rows)
    selected_label = ", ".join(sorted({r["data_label"] for r in selected_runs})) or ("demo" if label == "DEMO" else "real")
    selected_split = ", ".join(sorted({r["split_name"] for r in selected_runs})) or "none"
    st.markdown(f"**Active data label:** {selected_label.upper()} · **Split:** {selected_split} · **Question result rows:** {len(selected_rows)}")
    st.subheader("Download exports")
    try:
        import pandas as pd
    except ImportError:
        st.error("Install pandas to prepare CSV exports.")
        return
    per_question = []
    for item in selected_rows:
        result = item["result"]
        per_question.append({"run_id": item["run_id"], "configuration_id": item["configuration_id"], "split": item["split_name"],
                             "data_label": item["data_label"], "question_id": item["question_id"], "question": result.get("question"),
                             "answerable": result.get("answerable"), "recall": result.get("retrieval_metrics", {}).get("recall"),
                             "recall_at_5": result.get("retrieval_metrics_at_5", result.get("retrieval_metrics", {})).get("recall"),
                             "hit": result.get("retrieval_metrics", {}).get("hit"), "precision": result.get("retrieval_metrics", {}).get("precision"),
                             "mrr": result.get("retrieval_metrics", {}).get("mrr"), "ndcg": result.get("retrieval_metrics", {}).get("ndcg"),
                             "faithfulness": result.get("faithfulness"), "correctness_automated": result.get("correctness_automated"),
                             "unsupported_answer_rate": result.get("unsupported_answer_rate"), "latency_seconds": result.get("latency", {}).get("total_seconds")})
    aggregate_csv = pd.DataFrame(groups).drop(columns=["settings"], errors="ignore").to_csv(index=False)
    raw = store.export_json(selected_runs)
    if selected_label.lower() == "demo":
        validation_paths = {"demo": ("data/demo/demo_questions.jsonl", "data/demo/demo_pages.jsonl")}
    elif selected_split == "test":
        validation_paths = {"test": ("data/benchmark_test.jsonl", "data/processed/pages.jsonl")}
    elif selected_split == "dev":
        validation_paths = {"dev": ("data/benchmark_dev.jsonl", "data/processed/pages.jsonl")}
    else:
        validation_paths = {"dev": ("data/benchmark_dev.jsonl", "data/processed/pages.jsonl"),
                            "test": ("data/benchmark_test.jsonl", "data/processed/pages.jsonl")}
    validation_report = {}
    for split_name, (validation_path, pages_path) in validation_paths.items():
        pages, _ = _safe_read(pages_path, validate_page_record)
        questions, error = _safe_read(validation_path, validate_question_record)
        validation_report[split_name] = benchmark_counts(questions, pages, split_name, validation_path) if not error else {"valid": False, "error": error}
    snapshot = Path("results/frozen_test_config.yaml")
    frozen_data = snapshot.read_bytes() if snapshot.exists() else b""
    cols = st.columns(2)
    cols[0].download_button("Per-question results CSV", pd.DataFrame(per_question).to_csv(index=False), "per_question_results.csv", "text/csv", disabled=not per_question)
    cols[1].download_button("Aggregate metrics CSV", aggregate_csv, "aggregate_metrics.csv", "text/csv", disabled=not groups)
    cols = st.columns(3)
    cols[0].download_button("Full results and metadata JSON", json.dumps(raw, ensure_ascii=False, indent=2), "experiment_results.json", "application/json", disabled=not selected_runs)
    cols[1].download_button("Frozen experiment configuration YAML", frozen_data, "frozen_test_config.yaml", "application/yaml", disabled=not frozen_data)
    cols[2].download_button("Benchmark validation report JSON", json.dumps(validation_report, ensure_ascii=False, indent=2), "benchmark_validation.json", "application/json")


def render_app():
    import streamlit as st
    st.set_page_config(page_title="RAG Evidence Lab", page_icon="✦", layout="wide", initial_sidebar_state="expanded")
    _style(st)
    store = ExperimentStore("results/experiments.sqlite3")
    with st.sidebar:
        st.markdown('<div class="brand-lockup"><span class="brand-mark">R</span><div><div class="brand-name">RAG Evidence Lab</div><span class="brand-kicker">Evidence workspace</span></div></div>', unsafe_allow_html=True)
        page = st.radio("Navigation", NAV, key="navigation")
        label = st.selectbox("Active data source", ["REAL", "DEMO"], key="active_data_label")
        pages, chunks, questions, load_error = _active_data(st, label)
        docs = _document_rows(pages)
        st.divider()
        st.markdown("**Workspace status**")
        st.caption(f"Active corpus: {docs[0]['Title'] if len(docs)==1 else f'{len(docs)} indexed documents' if docs else 'No corpus'}")
        st.caption(f"Indexed documents: {len(docs)}")
        st.caption(f"Model API: {'Configured' if llm_configured() else 'Needs setup'}")
        st.caption(f"Displayed data: {label}")
        with st.expander("Settings"):
            chunk_sizes = sorted({chunk.chunk_size for chunk in chunks if chunk.chunk_size})
            st.caption("Chunk size: " + (", ".join(str(size) for size in chunk_sizes) + " tokens" if chunk_sizes else "Not indexed; default is 256 tokens"))
            st.caption(f"Embedding: {DEFAULT_EMBEDDING}")
            st.caption("Retrieval method: selected per query")
            st.caption("Top-k: selected per query")
    _render_header(st, page)
    if load_error and page in {"Overview", "Corpus", "Ask"}:
        st.warning(f"Some current data could not be loaded: {load_error}")
    if page == "Overview": _overview(st, pages, chunks, questions, label, store)
    elif page == "Corpus": _corpus_page(st, pages, chunks, label)
    elif page == "Ask": _ask_page(st, pages, chunks, label)
    elif page == "Experiments": _experiments_page(st, store, label)
    elif page == "Claims": _claim_page(st, store, label)
    elif page == "Retrieval debugger": _debugger_page(st, store, label)
    elif page == "Exports": _export_page(st, store, label)
