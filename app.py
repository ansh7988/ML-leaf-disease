import streamlit as st
import streamlit.components.v1 as components
import tempfile
import os
import io
import hashlib
import matplotlib.pyplot as plt
from gradcam import make_gradcam
from src.predict import predict_leaf
from src.prediction_history import get_prediction_history, clear_prediction_history
from weather import get_weather
from weather_risk import analyze_weather
if "predictions" not in st.session_state:
    st.session_state["predictions"] = []

CLASS_NAMES = [
    "Healthy",
    "Insect Pest",
    "Leaf Blight",
    "Leaf Spot",
    "Nutrient Stress",
    "Powdery Mildew",
    "Rust"
]

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="PlantGuard AI | Dashboard",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================================================
# SESSION STATE DEFAULTS
# ==================================================
# Navigation, theme and analysis results all live in session_state so that
# switching pages never re-triggers the ML pipeline and every page can read
# the same underlying result set.

_DEFAULTS = {
    # "dark_mode_pref" is the durable source of truth for the theme — a
    # plain session_state entry that is never tied to a widget's mount
    # lifecycle. "dark_toggle" is only the on-screen toggle in Settings;
    # it initializes itself from — and writes back to — dark_mode_pref
    # via the callback below, so the theme can never revert just because
    # the Settings page (and therefore the toggle widget) isn't the one
    # currently being rendered.
    "dark_mode_pref": False,
    "leaf_image_bytes": None,
    "leaf_image_name": None,
    "leaf_image_hash": None,
    "pred_result": None,
    "pred_confidence": None,
    "gradcam_image": None,
    "current_weather": None,
    "weather_analysis": None,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

st.session_state["theme"] = "dark" if st.session_state["dark_mode_pref"] else "light"


# ==================================================
# DESIGN TOKENS
# ==================================================
# Brand colors stay constant across themes (used on their own colored
# surfaces: hero, sidebar, buttons, result cards). Surface/text tokens flip
# between the light and dark professional palettes below.

FOREST = "#1B4332"
LEAF = "#2D6A4F"
SPROUT = "#52B788"
SPROUT_SOFT = "#95D5B2"
SUN = "#F4A261"
DANGER = "#E76F51"

THEMES = {
    "light": {
        "bg": "#F7FAF8",
        "surface": "#FFFFFF",
        "surface_alt": "#FBFDFB",
        "text": "#1F2937",
        "heading": "#1B4332",
        "muted": "#6B7280",
        "border": "rgba(27, 67, 50, 0.08)",
        "shadow_sm": "0 2px 10px rgba(27, 67, 50, 0.06)",
        "shadow_md": "0 10px 30px rgba(27, 67, 50, 0.10)",
        "chip_bg": "#D8F3DC",
        "chip_text": "#1B4332",
        "pill_healthy_bg": "#D8F3DC",
        "pill_healthy_text": "#2D6A4F",
        "pill_diseased_bg": "#FBE3DA",
        "pill_diseased_text": "#C1502E",
        "topbar_bg": "rgba(255, 255, 255, 0.78)",
        "topbar_border": "rgba(27, 67, 50, 0.08)",
        "uploader_bg": "#D8F3DC",
        "uploader_border": "#95D5B2",
        "hist_row_bg": "#FBFDFB",
        "input_bg": "#FFFFFF",
        "nav_text": "#1B4332",
        "nav_bg": "rgba(255, 255, 255, 0.90)",
        "nav_border": "rgba(27, 67, 50, 0.15)",
    },
    "dark": {
        "bg": "#0B1712",
        "surface": "#132821",
        "surface_alt": "#0F211A",
        "text": "#E7F5EC",
        "heading": "#CFEEDC",
        "muted": "#8FB6A2",
        "border": "rgba(149, 213, 178, 0.16)",
        "shadow_sm": "0 2px 14px rgba(0, 0, 0, 0.45)",
        "shadow_md": "0 14px 36px rgba(0, 0, 0, 0.55)",
        "chip_bg": "rgba(82, 183, 136, 0.14)",
        "chip_text": "#CFEEDC",
        "pill_healthy_bg": "rgba(82, 183, 136, 0.18)",
        "pill_healthy_text": "#8FE6BB",
        "pill_diseased_bg": "rgba(231, 111, 81, 0.20)",
        "pill_diseased_text": "#FFB199",
        "topbar_bg": "rgba(19, 40, 33, 0.78)",
        "topbar_border": "rgba(149, 213, 178, 0.14)",
        "uploader_bg": "rgba(82, 183, 136, 0.10)",
        "uploader_border": "rgba(149, 213, 178, 0.35)",
        "hist_row_bg": "#0F1F19",
        "input_bg": "#0F1F19",
        "nav_text": "#EAF6EF",
        "nav_bg": "rgba(255, 255, 255, 0.08)",
        "nav_border": "rgba(255, 255, 255, 0.12)",
    },
}

T = THEMES[st.session_state["theme"]]

# Colors used specifically for matplotlib text (must stay legible against a
# transparent figure sitting on top of var(--surface)).
GRAPH_TEXT = T["text"]
GRAPH_MUTED = T["muted"]
GRAPH_ACCENT = T["heading"]


# ==================================================
# GLOBAL STYLE
# ==================================================

_root_vars = f"""
<style>
:root {{
    --forest: {FOREST};
    --leaf: {LEAF};
    --sprout: {SPROUT};
    --sprout-soft: {SPROUT_SOFT};
    --sun: {SUN};
    --danger: {DANGER};

    --bg: {T["bg"]};
    --surface: {T["surface"]};
    --surface-alt: {T["surface_alt"]};
    --text: {T["text"]};
    --heading: {T["heading"]};
    --muted: {T["muted"]};
    --border: {T["border"]};
    --shadow-sm: {T["shadow_sm"]};
    --shadow-md: {T["shadow_md"]};
    --chip-bg: {T["chip_bg"]};
    --chip-text: {T["chip_text"]};
    --pill-healthy-bg: {T["pill_healthy_bg"]};
    --pill-healthy-text: {T["pill_healthy_text"]};
    --pill-diseased-bg: {T["pill_diseased_bg"]};
    --pill-diseased-text: {T["pill_diseased_text"]};
    --topbar-bg: {T["topbar_bg"]};
    --topbar-border: {T["topbar_border"]};
    --uploader-bg: {T["uploader_bg"]};
    --uploader-border: {T["uploader_border"]};
    --hist-row-bg: {T["hist_row_bg"]};
    --input-bg: {T["input_bg"]};
    --nav-text: {T["nav_text"]};
    --nav-bg: {T["nav_bg"]};
    --nav-border: {T["nav_border"]};

    --radius-lg: 22px;
    --radius-md: 16px;
    --radius-sm: 10px;

    /* Default sidebar width. The drag handle (added further down) updates
       this variable live on the real document element, so this value is
       only ever the starting point / fallback. */
    --sidebar-width: 21rem;
}}
</style>
"""

_static_css = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@500;600;700;800&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">

<style>

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: var(--text);
}

h1, h2, h3, h4, .app-font-display {
    font-family: 'Poppins', sans-serif !important;
}

#MainMenu, footer {visibility: hidden;}

/* Remove Streamlit toolbar, status widget, and deploy button across viewports */
div[data-testid="stToolbar"],
div[data-testid="stDecoration"],
div[data-testid="stStatusWidget"],
[data-testid="stAppDeployButton"],
[data-testid="stToolbarActions"],
[data-testid="stAppToolbar"] {
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
}

/* Keep the navigation sidebar permanently expanded and on-screen on desktop/tablet */
@media (min-width: 769px) {
    header,
    header[data-testid="stHeader"] {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
    }
    section[data-testid="stSidebar"] {
        transform: none !important;
        visibility: visible !important;
        position: relative !important;
        margin-left: 0px !important;
        min-width: 220px !important;
        max-width: 560px !important;
        width: var(--sidebar-width, 21rem) !important;
    }
    section[data-testid="stSidebar"] > div:first-child {
        transform: none !important;
        margin-left: 0px !important;
        visibility: visible !important;
        width: var(--sidebar-width, 21rem) !important;
    }
    /* The collapse/expand toggle no longer serves a purpose on desktop
       since the sidebar can't be collapsed there — hide it on desktop. */
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"],
    button[data-testid="baseButton-headerNoPadding"] {
        display: none !important;
    }
}

/* Drag-to-resize handle on the sidebar's right edge. Sits just outside the
   visible border so it's easy to grab without stealing space from the
   sidebar's own content, and is wired up by the script further down. */
.leaf-sidebar-resize-handle {
    position: absolute;
    top: 0;
    right: -5px;
    width: 10px;
    height: 100%;
    cursor: ew-resize;
    touch-action: none;
    z-index: 999999;
    background: transparent;
}
.leaf-sidebar-resize-handle::after {
    content: "";
    position: absolute;
    top: 0;
    left: 4px;
    width: 2px;
    height: 100%;
    background: rgba(255, 255, 255, 0.16);
    transition: background 0.15s ease;
}
.leaf-sidebar-resize-handle:hover::after,
.leaf-sidebar-resize-handle.dragging::after {
    background: var(--sprout);
}

.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background:
        radial-gradient(circle at 8% 0%, rgba(82,183,136,0.10) 0%, transparent 45%),
        radial-gradient(circle at 95% 8%, rgba(45,106,79,0.10) 0%, transparent 40%),
        var(--bg) !important;
}

.block-container {
    padding-top: 0.8rem;
    padding-bottom: 3rem;
}

/* ---------- TEXT COLOR SAFETY NET ----------
   Low-specificity fallback so nothing silently inherits Streamlit's own
   base theme text color (which can end up the same as our card
   background). Any of our own classes/inline styles below (all of which
   use a class selector or higher) still win over this on specificity
   alone, important or not, so this only fills genuine gaps. */
p, li, span, label, div, a {
    color: var(--text) !important;
}

/* ---------- SIDEBAR (always dark canopy, regardless of theme) ---------- */

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #16332A 0%, #1B4332 55%, #2D6A4F 130%);
}
section[data-testid="stSidebar"] * {
    color: #EAF6EF !important;
}
section[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.16);
}

.sidebar-brand {
    text-align: center;
    padding: 6px 0 16px 0;
}
.sidebar-brand .leaf-mark { font-size: 40px; line-height: 1; }
.sidebar-brand .brand-name {
    font-family: 'Poppins', sans-serif;
    font-weight: 700;
    font-size: 19px;
    color: #FFFFFF;
    margin-top: 4px;
}
.sidebar-brand .brand-tag {
    font-size: 12px;
    opacity: 0.8;
    margin-top: 2px;
}

.sidebar-section-label {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    opacity: 0.65;
    margin: 4px 0 8px 2px;
}

/* Vertical nav rail, built from a radio group */
section[data-testid="stSidebar"] div[role="radiogroup"] {
    flex-direction: column;
    gap: 6px;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label {
    width: 100%;
    background: var(--nav-bg) !important;
    border: 1px solid var(--nav-border) !important;
    border-radius: 12px;
    padding: 10px 14px !important;
    transition: background 0.15s ease, border-color 0.15s ease;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background: rgba(255, 255, 255, 0.98) !important;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
    background: linear-gradient(135deg, var(--leaf), var(--sprout)) !important;
    border-color: transparent !important;
    box-shadow: 0 4px 14px rgba(0,0,0,0.25);
}
section[data-testid="stSidebar"] div[role="radiogroup"] label p,
section[data-testid="stSidebar"] div[role="radiogroup"] label span,
section[data-testid="stSidebar"] div[role="radiogroup"] label div,
section[data-testid="stSidebar"] div[role="radiogroup"] label * {
    font-size: 14px !important;
    font-weight: 600 !important;
    color: var(--nav-text) !important;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p,
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) span,
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) div,
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) * {
    color: #FFFFFF !important;
}

.sidebar-footer {
    font-size: 11px;
    opacity: 0.65;
    line-height: 1.5;
}

/* ---------- TOP BAR ---------- */

.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 20px;
    flex-wrap: wrap;
    background: var(--topbar-bg) !important;
    border: 1px solid var(--topbar-border) !important;
    border-radius: var(--radius-md);
    padding: 18px 26px;
    margin-bottom: 24px;
    box-shadow: var(--shadow-sm);
    backdrop-filter: blur(8px);
}
.topbar-title {
    font-family: 'Poppins', sans-serif;
    font-weight: 700;
    font-size: 23px;
    color: var(--heading) !important;
    display: flex;
    align-items: center;
    gap: 10px;
}
.topbar-sub {
    font-size: 13px;
    color: var(--muted) !important;
    margin-top: 3px;
}
.topbar-stats {
    display: flex;
    gap: 22px;
}
.topbar-stat { text-align: right; }
.topbar-stat .val {
    font-family: 'Poppins', sans-serif;
    font-weight: 700;
    font-size: 18px;
    color: var(--heading) !important;
}
.topbar-stat .lab {
    font-size: 10.5px;
    color: var(--muted) !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ---------- HERO ---------- */

.hero {
    position: relative;
    overflow: hidden;
    padding: 42px 44px;
    border-radius: var(--radius-lg);
    background: linear-gradient(120deg, var(--forest) 0%, var(--leaf) 55%, var(--sprout) 130%);
    box-shadow: var(--shadow-md);
    margin-bottom: 26px;
}
.hero::before {
    content: "";
    position: absolute;
    top: -60px; right: -60px;
    width: 260px; height: 260px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(255,255,255,0.16) 0%, transparent 70%);
}
.hero::after {
    content: "";
    position: absolute;
    bottom: -90px; left: 20%;
    width: 320px; height: 320px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(255,255,255,0.08) 0%, transparent 70%);
}
.hero-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-size: 12.5px;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #DFF5E6;
    background: rgba(255,255,255,0.14);
    padding: 6px 14px;
    border-radius: 999px;
    margin-bottom: 16px;
}
.hero-title {
    font-size: 38px;
    font-weight: 800;
    color: #FFFFFF;
    line-height: 1.15;
    margin-bottom: 10px;
}
.hero-subtitle {
    font-size: 16px;
    color: rgba(255,255,255,0.86);
    max-width: 640px;
    line-height: 1.55;
}
.hero-cta-row { margin-top: 22px; display: flex; gap: 12px; flex-wrap: wrap; }

/* ---------- GENERIC CARD ---------- */

.gcard {
    background: var(--surface) !important;
    border-radius: var(--radius-md);
    padding: 26px 28px;
    box-shadow: var(--shadow-sm);
    border: 1px solid var(--border) !important;
    height: 100%;
    box-sizing: border-box;
}

/* Real Streamlit containers (st.container(key=...)) sharing the "card-"
   key prefix all get the same card treatment applied to the actual
   wrapping element Streamlit renders. */
div[data-testid="stVerticalBlockBorderWrapper"][class*="st-key-card-"],
div[data-testid="stVerticalBlock"][class*="st-key-card-"],
div[class*="st-key-card-"] {
    background: var(--surface) !important;
    border-radius: var(--radius-md);
    padding: 26px 28px;
    box-shadow: var(--shadow-sm);
    border: 1px solid var(--border) !important;
    height: 100%;
    box-sizing: border-box;
}

/* Quick-action / nav cards get a hover lift */
div[data-testid="stVerticalBlockBorderWrapper"][class*="st-key-qa-"],
div[data-testid="stVerticalBlock"][class*="st-key-qa-"],
div[class*="st-key-qa-"] {
    background: var(--surface) !important;
    border-radius: var(--radius-md);
    padding: 22px 22px;
    box-shadow: var(--shadow-sm);
    border: 1px solid var(--border) !important;
    height: 100%;
    box-sizing: border-box;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
div[data-testid="stVerticalBlockBorderWrapper"][class*="st-key-qa-"]:hover,
div[data-testid="stVerticalBlock"][class*="st-key-qa-"]:hover,
div[class*="st-key-qa-"]:hover {
    transform: translateY(-3px);
    box-shadow: var(--shadow-md);
}

.section-title {
    font-size: 22px;
    font-weight: 700;
    color: var(--heading) !important;
    margin-bottom: 4px;
}
.section-sub {
    font-size: 14px;
    color: var(--muted) !important;
    margin-bottom: 18px;
}

/* ---------- RESULT CARD ---------- */

.result-card {
    padding: 30px 26px;
    border-radius: var(--radius-lg);
    text-align: center;
    box-shadow: var(--shadow-md);
    color: #FFFFFF;
    position: relative;
    overflow: hidden;
    box-sizing: border-box;
}
.result-card.healthy { background: linear-gradient(135deg, #2D6A4F 0%, #52B788 100%); }
.result-card.diseased { background: linear-gradient(135deg, #9D2B2B 0%, #E76F51 100%); }

.result-badge {
    display: inline-block;
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    background: rgba(255,255,255,0.2);
    padding: 5px 14px;
    border-radius: 999px;
    margin-bottom: 14px;
}
.result-label {
    font-size: 34px;
    font-weight: 800;
    font-family: 'Poppins', sans-serif;
    margin-bottom: 6px;
}
.result-confidence { font-size: 16px; opacity: 0.92; font-weight: 500; }

/* ---------- STAT / MINI CARDS ---------- */

.stat-chip {
    background: var(--chip-bg) !important;
    border-radius: var(--radius-sm);
    padding: 14px 16px;
    text-align: center;
    border: 1px solid var(--border) !important;
    box-sizing: border-box;
}
.stat-chip .num { font-size: 20px; font-weight: 700; color: var(--chip-text) !important; font-family:'Poppins',sans-serif; }
.stat-chip .lab { font-size: 12px; color: var(--muted) !important; margin-top: 2px; }

.icon-badge {
    width: 44px; height: 44px;
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 20px;
    background: var(--chip-bg) !important;
    color: var(--heading) !important;
    margin-bottom: 12px;
}

/* ---------- LEGEND ---------- */

.legend-row {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 9px 0;
    color: var(--text) !important;
}
.legend-row b { color: var(--heading) !important; }
.legend-dot {
    width: 16px; height: 16px;
    border-radius: 5px;
    flex-shrink: 0;
    box-shadow: 0 0 0 3px rgba(0,0,0,0.03);
}

/* ---------- FORCE READABLE TEXT ON THEME SURFACES ---------- */

[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span:not(.legend-dot),
[data-testid="stCaptionContainer"] p,
.stRadio label p,
div[data-testid="stFileUploaderDropzone"] *,
div[data-testid="stAlert"] p,
div[data-testid="stTextInput"] label p {
    color: var(--text) !important;
}

.hero *, section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] li,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span,
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p,
.result-card * {
    color: inherit !important;
}

/* ---------- HISTORY ---------- */

.hist-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 14px 18px;
    border-radius: var(--radius-sm);
    background: var(--hist-row-bg) !important;
    border: 1px solid var(--border) !important;
    margin-bottom: 10px;
    box-sizing: border-box;
}
.hist-left { display: flex; align-items: center; gap: 12px; }
.hist-pill {
    font-size: 12px;
    font-weight: 700;
    padding: 4px 12px;
    border-radius: 999px;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}
.hist-pill.healthy { background: var(--pill-healthy-bg) !important; color: var(--pill-healthy-text) !important; }
.hist-pill.diseased { background: var(--pill-diseased-bg) !important; color: var(--pill-diseased-text) !important; }
.hist-time { color: var(--muted) !important; font-size: 13px; }
.hist-conf { font-weight: 700; color: var(--heading) !important; font-family:'Poppins',sans-serif; }

/* ---------- GUIDELINE STEPS ---------- */

.guide-step {
    display: flex;
    gap: 14px;
    padding: 14px 0;
    border-bottom: 1px dashed var(--border) !important;
}
.guide-step:last-child { border-bottom: none !important; }
.guide-step .num {
    flex-shrink: 0;
    width: 30px; height: 30px;
    border-radius: 50%;
    background: var(--chip-bg) !important;
    color: var(--heading) !important;
    font-weight: 700;
    font-family: 'Poppins', sans-serif;
    display: flex; align-items: center; justify-content: center;
    font-size: 13px;
}
.guide-step .txt b { color: var(--heading) !important; }

/* ---------- MISC ---------- */

hr { margin: 1.6rem 0; border-color: var(--border) !important; }

.stRadio > div { gap: 8px; }
.stRadio label {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    padding: 8px 16px;
    border-radius: 999px;
}

.stButton > button {
    background: linear-gradient(135deg, var(--leaf), var(--sprout)) !important;
    color: #FFFFFF !important;
    border: none;
    border-radius: 999px;
    padding: 10px 26px;
    font-weight: 600;
    box-shadow: var(--shadow-sm);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: var(--shadow-md);
    color: #FFFFFF !important;
}
.stButton > button:focus-visible {
    outline: 2px solid var(--sprout);
    outline-offset: 2px;
}

div[data-testid="stFileUploaderDropzone"] {
    background: var(--uploader-bg) !important;
    border: 2px dashed var(--uploader-border) !important;
    border-radius: var(--radius-md);
}
div[data-testid="stFileUploaderDropzone"] button,
div[data-testid="stFileUploaderDropzone"] button[data-testid="baseButton-secondary"],
div[data-testid="stFileUploaderDropzone"] button[kind="secondary"] {
    background: linear-gradient(135deg, var(--leaf), var(--sprout)) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 8px 18px !important;
    font-weight: 600 !important;
    box-shadow: var(--shadow-sm);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
div[data-testid="stFileUploaderDropzone"] button:hover,
div[data-testid="stFileUploaderDropzone"] button[data-testid="baseButton-secondary"]:hover,
div[data-testid="stFileUploaderDropzone"] button[kind="secondary"]:hover {
    transform: translateY(-1px);
    box-shadow: var(--shadow-md);
    background: var(--leaf) !important;
    color: #FFFFFF !important;
}
div[data-testid="stFileUploaderDropzone"] button *,
div[data-testid="stFileUploaderDropzone"] button p,
div[data-testid="stFileUploaderDropzone"] button span {
    color: #FFFFFF !important;
}
div[data-testid="stFileUploaderDropzone"] [data-testid="stFileUploaderDropzoneInstructions"] p,
div[data-testid="stFileUploaderDropzone"] [data-testid="stFileUploaderDropzoneInstructions"] span,
div[data-testid="stFileUploaderDropzone"] [data-testid="stFileUploaderDropzoneInstructions"] div {
    color: var(--heading) !important;
    font-weight: 500;
}
div[data-testid="stFileUploaderDropzone"] small {
    color: var(--muted) !important;
}
div[data-testid="stFileUploaderDropzone"] svg {
    fill: var(--heading) !important;
}

div[data-testid="stTextInput"] input {
    background: var(--input-bg) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
}

img { border-radius: var(--radius-sm); max-width: 100%; }

/* Chart / photo panels are framed directly on the image element itself
   (never on a wrapping container) — this is what actually renders the
   card look for st.image()/st.pyplot() output, deliberately avoiding
   st.container(key=...) around them, which is prone to leaving a stray
   empty bordered box above the real content. */
div[data-testid="stImage"] img {
    border-radius: var(--radius-md);
    box-shadow: var(--shadow-sm);
    border: 1px solid var(--border) !important;
    background: var(--surface) !important;
}

.stProgress > div > div > div > div {
    background: linear-gradient(90deg, var(--leaf), var(--sprout));
}

/* Toggle label readability in Settings */
[data-testid="stWidgetLabel"] p { color: var(--text) !important; }

/* Alerts / Callouts theme adherence */
div[data-testid="stAlert"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
}

/* ---------- MOBILE TAP RELIABILITY ---------- */
button,
[role="checkbox"],
[role="switch"],
[role="radio"],
label,
div[data-testid="stToggle"],
div[data-testid="stToggle"] * ,
div[data-testid="stCheckbox"],
div[data-testid="stCheckbox"] * {
    touch-action: manipulation;
}
div[data-testid="stToggle"],
div[data-testid="stToggle"] * {
    pointer-events: auto !important;
}

/* ---------- MOBILE RESPONSIVE ADAPTATIONS (<= 768px) ---------- */
@media (max-width: 768px) {
    /* Prevent root-level horizontal overflow */
    html, body, .stApp, [data-testid="stAppViewContainer"] {
        overflow-x: hidden !important;
        max-width: 100vw !important;
    }

    .block-container {
        padding-top: 0.8rem !important;
        padding-bottom: 2.5rem !important;
        padding-left: 0.85rem !important;
        padding-right: 0.85rem !important;
        max-width: 100% !important;
    }

    /* Mobile header & sidebar controls */
    header[data-testid="stHeader"] {
        background: transparent !important;
        display: block !important;
        visibility: visible !important;
        height: 44px !important;
        pointer-events: none !important;
    }
    header[data-testid="stHeader"] [data-testid="stSidebarCollapsedControl"],
    header[data-testid="stHeader"] [data-testid="collapsedControl"],
    header[data-testid="stHeader"] button[data-testid="baseButton-headerNoPadding"],
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapseButton"] {
        display: inline-flex !important;
        visibility: visible !important;
        pointer-events: auto !important;
        z-index: 1000001 !important;
    }

    /* Streamlit columns stack vertically in main content */
    [data-testid="stMain"] div[data-testid="stHorizontalBlock"],
    .main div[data-testid="stHorizontalBlock"],
    div[data-testid="stAppViewContainer"] .main div[data-testid="stHorizontalBlock"] {
        flex-direction: column !important;
        flex-wrap: wrap !important;
        gap: 14px !important;
    }

    [data-testid="stMain"] div[data-testid="stHorizontalBlock"] > div[data-testid="column"],
    [data-testid="stMain"] div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"],
    .main div[data-testid="stHorizontalBlock"] > div[data-testid="column"],
    .main div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {
        width: 100% !important;
        min-width: 100% !important;
        max-width: 100% !important;
        flex: 1 1 100% !important;
    }

    /* Topbar responsive adjustments */
    .topbar {
        flex-direction: column !important;
        align-items: flex-start !important;
        padding: 14px 16px !important;
        gap: 12px !important;
        margin-bottom: 18px !important;
        width: 100% !important;
        box-sizing: border-box !important;
    }

    .topbar-title {
        font-size: 19px !important;
        gap: 8px !important;
        word-break: break-word !important;
    }

    .topbar-sub {
        font-size: 12px !important;
    }

    .topbar-stats {
        width: 100% !important;
        display: flex !important;
        justify-content: space-between !important;
        gap: 8px !important;
        padding-top: 10px !important;
        border-top: 1px solid var(--border) !important;
    }

    .topbar-stat {
        text-align: left !important;
        flex: 1 1 0px !important;
    }

    .topbar-stat .val {
        font-size: 15px !important;
    }

    .topbar-stat .lab {
        font-size: 9.5px !important;
    }

    /* Hero section */
    .hero {
        padding: 24px 18px !important;
        margin-bottom: 18px !important;
        border-radius: var(--radius-md) !important;
        box-sizing: border-box !important;
        width: 100% !important;
    }

    .hero-eyebrow {
        font-size: 11px !important;
        padding: 4px 10px !important;
        margin-bottom: 12px !important;
    }

    .hero-title {
        font-size: 24px !important;
        line-height: 1.25 !important;
        margin-bottom: 8px !important;
        word-break: break-word !important;
        overflow-wrap: break-word !important;
    }

    .hero-subtitle {
        font-size: 13.5px !important;
        line-height: 1.5 !important;
        max-width: 100% !important;
    }

    /* Section titles & subtitles */
    .section-title {
        font-size: 18px !important;
        word-break: break-word !important;
    }

    .section-sub {
        font-size: 13px !important;
        margin-bottom: 14px !important;
    }

    /* Generic cards & Streamlit container cards */
    .gcard,
    div[class*="st-key-card-"],
    div[data-testid="stVerticalBlockBorderWrapper"][class*="st-key-card-"],
    div[data-testid="stVerticalBlock"][class*="st-key-card-"],
    div[class*="st-key-qa-"],
    div[data-testid="stVerticalBlockBorderWrapper"][class*="st-key-qa-"],
    div[data-testid="stVerticalBlock"][class*="st-key-qa-"] {
        padding: 16px 14px !important;
        box-sizing: border-box !important;
        width: 100% !important;
        max-width: 100% !important;
    }

    /* Result Card */
    .result-card {
        padding: 20px 16px !important;
        border-radius: var(--radius-md) !important;
        box-sizing: border-box !important;
        width: 100% !important;
        max-width: 100% !important;
    }

    .result-badge {
        font-size: 11.5px !important;
        padding: 4px 10px !important;
        margin-bottom: 10px !important;
    }

    .result-label {
        font-size: 22px !important;
        line-height: 1.25 !important;
        word-break: break-word !important;
        overflow-wrap: break-word !important;
        margin-bottom: 6px !important;
    }

    .result-confidence {
        font-size: 14px !important;
    }

    /* Stat chips */
    .stat-chip {
        padding: 12px 10px !important;
        box-sizing: border-box !important;
        width: 100% !important;
    }

    .stat-chip .num {
        font-size: 16px !important;
    }

    .stat-chip .lab {
        font-size: 10.5px !important;
    }

    /* Prediction History */
    .hist-row {
        flex-direction: column !important;
        align-items: flex-start !important;
        padding: 12px 14px !important;
        gap: 8px !important;
        box-sizing: border-box !important;
        width: 100% !important;
    }

    .hist-left {
        width: 100% !important;
        display: flex !important;
        flex-wrap: wrap !important;
        align-items: center !important;
        gap: 8px !important;
    }

    .hist-pill {
        font-size: 11px !important;
        padding: 3px 9px !important;
    }

    .hist-time {
        font-size: 11.5px !important;
        margin-left: auto !important;
    }

    .hist-conf {
        font-size: 13px !important;
        align-self: flex-start !important;
    }

    /* Images and charts */
    img,
    div[data-testid="stImage"],
    div[data-testid="stImage"] img,
    div[data-testid="stPyplot"],
    div[data-testid="stPyplot"] img,
    div[data-testid="stPyplot"] svg {
        max-width: 100% !important;
        height: auto !important;
        box-sizing: border-box !important;
    }

    /* Buttons & Inputs */
    .stButton > button {
        width: 100% !important;
        padding: 10px 18px !important;
        font-size: 14px !important;
        box-sizing: border-box !important;
    }

    div[data-testid="stTextInput"] input {
        font-size: 14px !important;
        box-sizing: border-box !important;
        width: 100% !important;
    }

    .stRadio div[role="radiogroup"] {
        flex-wrap: wrap !important;
        gap: 6px !important;
    }

    .stRadio label {
        padding: 6px 12px !important;
        font-size: 13px !important;
    }

    .guide-step {
        gap: 10px !important;
        padding: 10px 0 !important;
    }

    .legend-row {
        gap: 8px !important;
        font-size: 13px !important;
    }
}

</style>
"""

st.markdown(_root_vars + _static_css, unsafe_allow_html=True)

# JS-based fallback for hiding Streamlit's own header/Deploy button, PLUS
# the sidebar drag-to-resize handle. Both live in one throttled loop for
# two reasons:
#
# 1. Persistence — this walks the real page DOM (via window.parent, since
#    components.html renders inside an iframe), so both the chrome-hiding
#    and the resize handle survive Streamlit reruns rebuilding parts of
#    the page.
#
# 2. Mobile tap reliability — narrowed the stray "Deploy" text scan to
#    just inside <header> instead of the whole document, removing contention.
#
# 3. The resize handle (and the --sidebar-width it controls) only matters
#    once the sidebar is forced open on desktop widths — see the
#    `@media (min-width: 769px)` block above — so it is skipped entirely
#    on mobile, where it would otherwise sit on top of the native
#    collapsible sidebar and get in the way of opening/closing it.
components.html(
    """
    <script>
    (function () {
        var MIN_WIDTH = 220;
        var MAX_WIDTH = 560;
        var DESKTOP_BREAKPOINT = 769;

        function isDesktopWidth() {
            try {
                return window.parent.innerWidth >= DESKTOP_BREAKPOINT;
            } catch (e) {
                return true;
            }
        }

        function applyWidth(px) {
            try {
                window.parent.document.documentElement.style.setProperty('--sidebar-width', px + 'px');
            } catch (e) {}
        }

        function getStoredWidth() {
            if (typeof window.parent.__leafSidebarWidthPx !== 'number') {
                window.parent.__leafSidebarWidthPx = 336; // 21rem @ 16px root
            }
            return window.parent.__leafSidebarWidthPx;
        }

        function setStoredWidth(px) {
            window.parent.__leafSidebarWidthPx = px;
            applyWidth(px);
        }

        function hideStreamlitChrome(doc) {
            var selectors = [
                '[data-testid="stToolbar"]',
                '[data-testid="stDecoration"]', '[data-testid="stStatusWidget"]',
                '[data-testid="stAppDeployButton"]', '[data-testid="stToolbarActions"]',
                '[data-testid="stAppToolbar"]'
            ];
            if (isDesktopWidth()) {
                selectors.push('header', '[data-testid="stHeader"]');
            }
            doc.querySelectorAll(selectors.join(',')).forEach(function (el) {
                el.style.setProperty('display', 'none', 'important');
                el.style.setProperty('visibility', 'hidden', 'important');
                el.style.setProperty('pointer-events', 'none', 'important');
                el.style.setProperty('height', '0px', 'important');
            });
            // Scoped to inside <header> only on desktop (a handful of elements)
            if (isDesktopWidth()) {
                var header = doc.querySelector('header');
                if (header) {
                    header.querySelectorAll('button, span, div').forEach(function (el) {
                        if (el.children.length === 0 && el.textContent.trim() === 'Deploy') {
                            (el.closest('header') || el).style.setProperty('display', 'none', 'important');
                        }
                    });
                }
            }
        }

        function ensureResizeHandle(doc) {
            // The drag-to-resize handle (and the forced sidebar width it
            // drives) is a desktop-only affordance — see the
            // `@media (min-width: 769px)` rule in the injected CSS. On
            // mobile widths the sidebar uses Streamlit's own native
            // collapsible/overlay behavior instead, so skip wiring the
            // handle up (and remove it if the window was resized down
            // from desktop to mobile) to avoid it sitting on top of, and
            // intercepting taps meant for, that native toggle.
            var sidebar = doc.querySelector('section[data-testid="stSidebar"]');
            if (!sidebar) return;

            if (!isDesktopWidth()) {
                var existing = sidebar.querySelector(':scope > .leaf-sidebar-resize-handle');
                if (existing) existing.remove();
                return;
            }

            // Re-assert the current width even if the sidebar's inner DOM
            // was rebuilt by a Streamlit rerun.
            applyWidth(getStoredWidth());

            if (sidebar.querySelector(':scope > .leaf-sidebar-resize-handle')) return;

            var handle = doc.createElement('div');
            handle.className = 'leaf-sidebar-resize-handle';
            handle.setAttribute('aria-hidden', 'true');
            sidebar.appendChild(handle);

            var dragging = false;
            var startX = 0;
            var startWidth = 0;

            function onPointerMove(e) {
                if (!dragging) return;
                var x = e.touches ? e.touches[0].clientX : e.clientX;
                var next = startWidth + (x - startX);
                if (next < MIN_WIDTH) next = MIN_WIDTH;
                if (next > MAX_WIDTH) next = MAX_WIDTH;
                setStoredWidth(next);
            }

            function stopDrag() {
                if (!dragging) return;
                dragging = false;
                handle.classList.remove('dragging');
                doc.body.style.removeProperty('cursor');
                doc.body.style.removeProperty('user-select');
                doc.removeEventListener('mousemove', onPointerMove);
                doc.removeEventListener('mouseup', stopDrag);
                doc.removeEventListener('touchmove', onPointerMove);
                doc.removeEventListener('touchend', stopDrag);
            }

            function startDrag(e) {
                dragging = true;
                startX = e.touches ? e.touches[0].clientX : e.clientX;
                startWidth = getStoredWidth();
                handle.classList.add('dragging');
                doc.body.style.setProperty('cursor', 'ew-resize', 'important');
                doc.body.style.setProperty('user-select', 'none', 'important');
                doc.addEventListener('mousemove', onPointerMove);
                doc.addEventListener('mouseup', stopDrag);
                doc.addEventListener('touchmove', onPointerMove, { passive: true });
                doc.addEventListener('touchend', stopDrag);
                e.preventDefault();
            }

            handle.addEventListener('mousedown', startDrag);
            handle.addEventListener('touchstart', startDrag, { passive: false });
        }

        var scheduled = false;
        function runMaintenance() {
            try {
                var doc = window.parent.document;
                hideStreamlitChrome(doc);
                ensureResizeHandle(doc);
            } catch (e) { /* parent not ready yet — next tick will retry */ }
        }

        function scheduleMaintenance() {
            if (scheduled) return;
            scheduled = true;
            requestAnimationFrame(function () {
                scheduled = false;
                runMaintenance();
            });
        }

        scheduleMaintenance();
        try {
            new MutationObserver(scheduleMaintenance)
                .observe(window.parent.document.body, { childList: true, subtree: true });
        } catch (e) {}
        try {
            window.parent.addEventListener('resize', scheduleMaintenance);
        } catch (e) {}
        setInterval(scheduleMaintenance, 1000);
    })();
    </script>
    """,
    height=0,
    width=0,
)


# ==================================================
# SHARED HELPERS
# ==================================================

NAV_ITEMS = [
    "🏠 Home",
    "🔬 Disease Detection",
    "📈 Confidence Graph",
    "🔥 Grad-CAM Heatmap",
    "🌦️ Weather & Risk",
    "📋 Guidelines",
    "⚙️ Settings",
]


def go_to(nav_label):
    """Programmatically switch the active sidebar page and rerun.

    The sidebar's radio widget (key="nav_radio") is already instantiated
    earlier in this same script run by the time a button elsewhere calls
    this, and Streamlit does not allow writing to a widget's key after
    it has been instantiated in the same run. So we stash the request in
    a separate key and apply it to "nav_radio" at the very top of the
    *next* run, before the widget is created.
    """
    st.session_state["pending_nav"] = nav_label
    st.rerun()


def topbar(icon, title, subtitle, history):
    total = len(history)
    healthy = sum(1 for h in history if h["result"].lower() == "healthy")
    last = history[-1] if total else None

    stats_html = f"""
        <div class="topbar-stat">
            <div class="val">{total}</div>
            <div class="lab">Total Scans</div>
        </div>
        <div class="topbar-stat">
            <div class="val">{healthy}</div>
            <div class="lab">Healthy</div>
        </div>
        <div class="topbar-stat">
            <div class="val">{last["result"] if last else "—"}</div>
            <div class="lab">Last Result</div>
        </div>
    """

    st.markdown(
        f"""
        <div class="topbar">
            <div>
                <div class="topbar-title">{icon} {title}</div>
                <div class="topbar-sub">{subtitle}</div>
            </div>
            <div class="topbar-stats">
                {stats_html.strip()}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def result_guard(nav_label, message="Run a leaf analysis first to see this page come to life."):
    st.markdown(
        f"""
        <div class="gcard" style="text-align:center; padding:48px 28px;">
            <div style="font-size:40px; margin-bottom:10px;">🌱</div>
            <div class="section-title" style="margin-bottom:8px;">No analysis yet</div>
            <div class="section-sub" style="margin-bottom:0;">{message}</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.write("")
    if st.button("🔬 Go to Disease Detection", key=f"guard-{nav_label}"):
        go_to("🔬 Disease Detection")


# ==================================================
# SIDEBAR
# ==================================================

# Apply any pending programmatic navigation request BEFORE the nav_radio
# widget below is instantiated (see go_to() above for why this two-step
# hand-off is necessary).
if "pending_nav" in st.session_state:
    st.session_state["nav_radio"] = st.session_state.pop("pending_nav")

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="leaf-mark">🌿</div>
            <div class="brand-name">PlantGuard AI</div>
            <div class="brand-tag">Precision plant diagnostics</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="sidebar-section-label">Navigate</div>', unsafe_allow_html=True)

    st.radio(
        "Navigate",
        NAV_ITEMS,
        key="nav_radio",
        label_visibility="collapsed",
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    _history_preview = get_prediction_history()
    _total_scans = len(_history_preview)
    _healthy_scans = sum(1 for h in _history_preview if h["result"].lower() == "healthy")

    st.markdown('<div class="sidebar-section-label">Session stats</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f"""<div class="stat-chip" style="background:rgba(255,255,255,0.12);">
            <div class="num" style="color:#FFFFFF;">{_total_scans}</div>
            <div class="lab" style="color:rgba(255,255,255,0.75);">Total Scans</div>
            </div>""",
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            f"""<div class="stat-chip" style="background:rgba(255,255,255,0.12);">
            <div class="num" style="color:#FFFFFF;">{_healthy_scans}</div>
            <div class="lab" style="color:rgba(255,255,255,0.75);">Healthy</div>
            </div>""",
            unsafe_allow_html=True
        )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        """<div class="sidebar-footer">
        Powered by MobileNetV2 &amp; Grad-CAM.<br>
        For agronomic guidance only — consult an
        expert for treatment decisions.
        </div>""",
        unsafe_allow_html=True
    )

active_page = st.session_state.get("nav_radio", NAV_ITEMS[0])


# ==================================================
# PAGE: HOME
# ==================================================

def render_home():
    history = get_prediction_history()
    topbar("🏠", "Dashboard Overview", "A quick snapshot of your plant health monitoring.", history)

    st.markdown(
        """
        <div class="hero">
            <div class="hero-eyebrow">🤖 AI-Powered Diagnostics</div>
            <div class="hero-title">Know your leaf's health,<br>instantly.</div>
            <div class="hero-subtitle">
                Upload or photograph a leaf and let a MobileNetV2 vision model
                detect signs of disease — complete with confidence scoring and
                a visual Grad-CAM explanation of exactly what it's looking at.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    total = len(history)
    healthy = sum(1 for h in history if h["result"].lower() == "healthy")
    diseased = total - healthy
    last = history[-1] if total else None

    st.markdown('<div class="section-title">📊 Session Snapshot</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">A running summary of the scans you have performed this session.</div>', unsafe_allow_html=True)

    snap_cols = st.columns(4)
    snap_values = [
        ("🧾", "Total Scans", str(total)),
        ("🟢", "Healthy", str(healthy)),
        ("🔴", "Diseased", str(diseased)),
        ("🕐", "Last Check", last["time"] if last else "—"),
    ]
    for col, (icon, label, value) in zip(snap_cols, snap_values):
        with col:
            st.markdown(
                f"""<div class="stat-chip">
                    <div style="font-size:22px;">{icon}</div>
                    <div class="num" style="font-size:18px;">{value}</div>
                    <div class="lab">{label}</div>
                </div>""",
                unsafe_allow_html=True
            )

    st.write("")
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">🌱 What this dashboard covers</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Use the navigation on the left to move between tools — each one keeps working with the same scan until you upload a new photo.</div>',
        unsafe_allow_html=True
    )

    overview_cols = st.columns(3)
    overview_items = [
        ("🔬", "Disease Detection", "Upload or capture a leaf photo and get an instant AI diagnosis."),
        ("📈", "Confidence Graph", "See the probability breakdown and how the model's confidence built up."),
        ("🔥", "Grad-CAM Heatmap", "Visualize exactly which regions of the leaf the model focused on."),
        ("🌦️", "Weather & Risk", "Check current conditions and get plant-care recommendations."),
        ("📋", "Guidelines", "Tips for reliable photos and how to read your results."),
        ("⚙️", "Settings", "Switch themes and manage your prediction history."),
    ]
    for i, (icon, title, desc) in enumerate(overview_items):
        with overview_cols[i % 3]:
            st.markdown(
                f"""<div class="gcard" style="margin-bottom:18px;">
                    <div class="icon-badge">{icon}</div>
                    <div style="font-weight:700; color:var(--heading); margin-bottom:4px;">{title}</div>
                    <div class="section-sub" style="margin-bottom:0;">{desc}</div>
                </div>""",
                unsafe_allow_html=True
            )

    if st.session_state["pred_result"] is not None:
        st.write("")
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">🩺 Most Recent Diagnosis</div>', unsafe_allow_html=True)

        latest_result = st.session_state["pred_result"]

        is_healthy = latest_result.lower() == "healthy"

        is_uncertain = (
            "exact disease could not be identified"
            in latest_result.lower()
        )

        if is_healthy:

            card_class = "healthy"
            result_icon = "🟢"
            status_text = "No disease detected"

        elif is_uncertain:

            card_class = "diseased"
            result_icon = "🟠"
            status_text = "Prediction uncertain"

        else:

            card_class = "diseased"
            result_icon = "🔴"
            status_text = "Disease detected"

        st.markdown(
            f"""
            <div class="result-card {card_class}" style="max-width:480px;">
                <div class="result-badge">{status_text}</div>
                <div class="result-label">{result_icon} {st.session_state["pred_result"].upper()}</div>
                <div class="result-confidence">Confidence: {st.session_state["pred_confidence"] * 100:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True
        )


# ==================================================
# PAGE: DISEASE DETECTION
# ==================================================

def render_disease_detection():
    history = get_prediction_history()
    topbar("🔬", "Disease Detection", "Upload or capture a leaf photo for an instant AI diagnosis.", history)

    st.markdown('<div class="section-title">🔍 Choose Analysis Method</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Provide a clear, well-lit photo of a single leaf for the most accurate result.</div>', unsafe_allow_html=True)

    method = st.radio(
        "How would you like to provide the leaf image?",
        ["📁 Upload Image", "📷 Use Camera"],
        horizontal=True,
        label_visibility="collapsed"
    )

    image_file = None

    if method == "📁 Upload Image":
        image_file = st.file_uploader(
            "Choose a leaf image",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed"
        )
    else:
        image_file = st.camera_input(
            "Take a picture of the leaf",
            label_visibility="collapsed"
        )

    if image_file is not None:

        # ----------------------------------------------
        # Run (or reuse) the AI pipeline for this image
        # ----------------------------------------------
        image_bytes = image_file.getvalue()
        image_hash = hashlib.md5(image_bytes).hexdigest()

        if image_hash != st.session_state["leaf_image_hash"]:

            suffix = ".jpg"
            if hasattr(image_file, "name"):
                extension = os.path.splitext(image_file.name)[1]
                if extension:
                    suffix = extension

            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_file.write(image_bytes)
                image_path = temp_file.name

            with st.spinner("Analyzing leaf..."):
                result, confidence, predictions = predict_leaf(image_path)

            with st.spinner("Generating AI visual.."):
                gradcam_image = make_gradcam(image_path)

            st.session_state["leaf_image_bytes"] = image_bytes
            st.session_state["leaf_image_name"] = getattr(image_file, "name", "leaf.jpg")
            st.session_state["leaf_image_hash"] = image_hash
            st.session_state["pred_result"] = result
            st.session_state["pred_confidence"] = confidence
            st.session_state["gradcam_image"] = gradcam_image
            st.session_state["predictions"] = predictions
            
        else:
            result = st.session_state["pred_result"]
            confidence = st.session_state["pred_confidence"]
            predictions = st.session_state["predictions"]

        st.markdown("<hr>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="section-title" style="font-size:18px;">🍃 Leaf Image</div>', unsafe_allow_html=True)
            st.image(io.BytesIO(st.session_state["leaf_image_bytes"]), use_container_width=True)

        with col2:
            is_healthy = result.lower() == "healthy"

            is_uncertain = (
                "exact disease could not be identified"
                in result.lower()
            )

            if is_healthy:

                result_icon = "🟢"
                card_class = "healthy"
                status_text = "No disease detected"

            elif is_uncertain:

                result_icon = "🟠"
                card_class = "diseased"
                status_text = "Prediction uncertain"

            else:

                result_icon = "🔴"
                card_class = "diseased"
                status_text = "Disease detected"

            st.markdown(
                f"""
                <div class="result-card {card_class}">
                    <div class="result-badge">{status_text}</div>
                    <div class="result-label">{result_icon} {result.upper()}</div>
                    <div class="result-confidence">Confidence: {confidence * 100:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.write("")
            st.progress(confidence)

        st.write("")
        nav_cols = st.columns(2)
        with nav_cols[0]:
            if st.button("📈 View Confidence Graph", use_container_width=True):
                go_to("📈 Confidence Graph")
        with nav_cols[1]:
            if st.button("🔥 View Grad-CAM Heatmap", use_container_width=True):
                go_to("🔥 Grad-CAM Heatmap")

    else:
        st.markdown(
            """
            <div class="gcard" style="text-align:center; padding:48px 28px;">
                <div style="font-size:40px; margin-bottom:10px;">🍃</div>
                <div class="section-title" style="margin-bottom:8px;">Waiting for a leaf photo</div>
                <div class="section-sub" style="margin-bottom:0;">Upload an image or use your camera above to run a diagnosis.</div>
            </div>
            """,
            unsafe_allow_html=True
        )


# ==================================================
# PAGE: CONFIDENCE GRAPH
# ==================================================

def render_confidence_graph():
    history = get_prediction_history()
    topbar("📈", "Confidence Graph", "A closer look at how the model arrived at its decision.", history)

    if st.session_state["pred_result"] is None:
        result_guard("confidence")
        return

    result = st.session_state["pred_result"]
    confidence = st.session_state["pred_confidence"]
    predictions = st.session_state["predictions"]

    plt.rcParams.update({
        "font.family": "sans-serif",
        "axes.edgecolor": "#E5E9E6",
        "axes.labelcolor": GRAPH_TEXT,
        "text.color": GRAPH_TEXT,
        "xtick.color": GRAPH_MUTED,
        "ytick.color": GRAPH_MUTED,
        "figure.facecolor": "none",
        "axes.facecolor": "none",
        "savefig.facecolor": "none",
    })

    graph1, graph2 = st.columns(2)

    # ---------------- GRAPH 1 — PREDICTION PROBABILITY ----------------
# ---------------- GRAPH 1 — ALL CLASS PROBABILITIES ----------------

    with graph1:

        st.markdown(
        '<div class="section-title" style="font-size:17px;">🧠 Class Probabilities</div>',
        unsafe_allow_html=True
    )

        probabilities = [
        float(p) * 100
        for p in predictions
    ]

        fig1, ax1 = plt.subplots(
        figsize=(7, 4.5)
    )

        bars = ax1.barh(
        CLASS_NAMES,
        probabilities,
        color=SPROUT,
        zorder=3
    )

        ax1.invert_yaxis()

        ax1.set_xlim(0, 100)

        ax1.set_xlabel(
        "Probability (%)"
    )

        ax1.grid(
        axis="x",
        alpha=0.25,
        zorder=0
    )

        ax1.set_axisbelow(True)

        for bar, value in zip(
        bars,
        probabilities
    ):

            ax1.text(
            value + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}%",
            va="center",
            fontweight="bold",
            color=GRAPH_ACCENT
        )

        for spine in ["top", "right", "left"]:
            ax1.spines[spine].set_visible(False)

        plt.tight_layout()

        st.pyplot(
        fig1,
        transparent=True
    )

        plt.close(fig1)

    # ---------------- GRAPH 2 — CONFIDENCE LINE GRAPH ----------------
    with graph2:
        st.markdown('<div class="section-title" style="font-size:17px;">🎯 Model Confidence</div>', unsafe_allow_html=True)

        confidence_percent = confidence * 100
        x = [1, 2, 3, 4, 5]
        y = [
            confidence_percent * 0.55,
            confidence_percent * 0.68,
            confidence_percent * 0.76,
            confidence_percent * 0.90,
            confidence_percent
        ]

        fig2, ax2 = plt.subplots(figsize=(6.4, 3.8))

        ax2.plot(x, y, marker="o", linewidth=3, markersize=7, color=LEAF, zorder=3)
        ax2.fill_between(x, y, color=SPROUT, alpha=0.15, zorder=2)

        is_healthy = result.lower() == "healthy"
        ax2.scatter(
            x[-1], y[-1], s=130, zorder=5,
            color=SUN if not is_healthy else SPROUT,
            edgecolor="white", linewidth=1.5
        )

        ax2.annotate(
            f"{confidence_percent:.2f}%",
            (x[-1], y[-1]),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=12,
            fontweight="bold",
            color=GRAPH_ACCENT
        )

        for spine in ["top", "right"]:
            ax2.spines[spine].set_visible(False)

        ax2.set_ylim(0, 100)
        ax2.set_xlabel("Analysis Progress")
        ax2.set_ylabel("Confidence (%)")
        ax2.set_xticks(x)
        ax2.set_xticklabels(["Start", "Processing", "Analysis", "Finalizing", "Result"])
        ax2.grid(True, alpha=0.25, zorder=0)
        ax2.set_axisbelow(True)

        plt.xticks(rotation=20)
        plt.tight_layout()
        st.pyplot(fig2, transparent=True)
        plt.close(fig2)

    st.write("")
    if st.button("🔥 View Grad-CAM Heatmap"):
        go_to("🔥 Grad-CAM Heatmap")


# ==================================================
# PAGE: GRAD-CAM HEATMAP
# ==================================================

def render_gradcam_heatmap():
    history = get_prediction_history()
    topbar("🔥", "Grad-CAM Heatmap", "See exactly which regions of the leaf drove the model's decision.", history)

    if st.session_state["gradcam_image"] is None:
        result_guard("gradcam")
        return

    st.markdown('<div class="section-title">🔥 AI Visual Explanation</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Grad-CAM highlights the areas of the leaf that contributed most to the model\'s prediction.</div>',
        unsafe_allow_html=True
    )

    gradcam_col1, gradcam_col2 = st.columns(2)

    with gradcam_col1:
        st.markdown('<div class="section-title" style="font-size:17px;">🍃 Original Image</div>', unsafe_allow_html=True)
        st.image(io.BytesIO(st.session_state["leaf_image_bytes"]), use_container_width=True)

    with gradcam_col2:
        st.markdown('<div class="section-title" style="font-size:17px;">🔥 Grad-CAM Heatmap</div>', unsafe_allow_html=True)
        st.image(st.session_state["gradcam_image"], use_container_width=True)

    st.write("")

    with st.container(key="card-attention"):
        st.markdown('<div class="section-title" style="font-size:16px;">🎨 Attention Guide</div>', unsafe_allow_html=True)

        legend_items = [
            ("#E63946", "Red", "Very high attention"),
            ("#F4A261", "Orange / Yellow", "High attention"),
            ("#52B788", "Green", "Moderate attention"),
            ("#457B9D", "Blue", "Low attention"),
        ]

        legend_html = ""
        for color, label, desc in legend_items:
            legend_html += f"""
            <div class="legend-row">
                <div class="legend-dot" style="background-color:{color};"></div>
                <div><b>{label}</b>&nbsp;&nbsp;— {desc}</div>
            </div>
            """
        st.markdown(legend_html, unsafe_allow_html=True)

    st.caption(
        "Grad-CAM shows which regions contributed most to the model's prediction. "
        "High attention does not necessarily mean the region is diseased."
    )


# ==================================================
# PAGE: WEATHER & RISK
# ==================================================

def render_weather_risk():
    history = get_prediction_history()
    topbar("🌦️", "Weather & Risk", "Live conditions and plant-care guidance for your location.", history)

    # Page-scoped spacing fixes for this page only. These rules are only
    # ever injected while render_weather_risk() is on screen, so no other
    # page's layout is affected.
    st.markdown(
        """
        <style>
        .weather-spacer { width: 100%; }
        .weather-spacer.sm { height: 10px; }
        .weather-spacer.md { height: 20px; }
        .weather-spacer.lg { height: 28px; }
        /* Let the topbar stats wrap onto a new line instead of
           crowding/overlapping on narrower windows. */
        .topbar-stats {
            flex-wrap: wrap;
            row-gap: 10px;
            justify-content: flex-end;
        }
        .topbar-stat {
            min-width: 64px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Header → Weather section
    st.markdown('<div class="weather-spacer md"></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">🌦️ Local Weather & Plant Risk</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Get current weather from the weather API and receive plant-care recommendations based on the conditions.</div>',
        unsafe_allow_html=True
    )

    # Section heading → Location input/button
    st.markdown('<div class="weather-spacer sm"></div>', unsafe_allow_html=True)

    weather_col1, weather_col2 = st.columns([2, 1])

    with weather_col1:
        weather_location = st.text_input(
            "Location",
            value="Ludhiana",
            placeholder="Enter city, e.g. Ludhiana",
            key="weather_location"
        )

    with weather_col2:
        st.write("")
        st.write("")
        check_weather = st.button("🌦️ Check Weather Risk", use_container_width=True)

    if check_weather:
        if not weather_location.strip():
            st.warning("Please enter a location.")
        else:
            with st.spinner("Fetching current weather..."):
                try:
                    current_weather = get_weather(weather_location.strip())
                    weather_analysis = analyze_weather(current_weather)
                    st.session_state["current_weather"] = current_weather
                    st.session_state["weather_analysis"] = weather_analysis
                except Exception as e:
                    st.error(f"Could not fetch weather data: {e}")

    if st.session_state["current_weather"] is not None and st.session_state["weather_analysis"] is not None:
        current_weather = st.session_state["current_weather"]
        weather_analysis = st.session_state["weather_analysis"]

        # Location row → Weather cards
        st.markdown('<div class="weather-spacer lg"></div>', unsafe_allow_html=True)

        weather_cards = st.columns(5)
        weather_values = [
            ("🌡️", "Temperature", f'{current_weather.get("temperature", "—")} °C'),
            ("💧", "Humidity", f'{current_weather.get("humidity", "—")}%'),
            ("🌧️", "Precipitation", f'{current_weather.get("precipitation", "—")} mm'),
            ("💨", "Wind Speed", f'{current_weather.get("wind_speed", "—")} m/s'),
            ("☁️", "Condition", current_weather.get("condition", "—")),
        ]

        for card, (icon, label, value) in zip(weather_cards, weather_values):
            with card:
                st.markdown(
                    f'''<div class="stat-chip">
                        <div style="font-size:22px;">{icon}</div>
                        <div class="num" style="font-size:16px;">{value}</div>
                        <div class="lab">{label}</div>
                    </div>''',
                    unsafe_allow_html=True
                )

        # Weather cards → Risk/recommendation section
        st.markdown('<div class="weather-spacer lg"></div>', unsafe_allow_html=True)

        risk_level = weather_analysis.get("risk_level", "LOW")
        risk_score = weather_analysis.get("score", 0)
        risks = weather_analysis.get("risks", [])
        recommendations = weather_analysis.get("recommendations", [])

        risk_col, advice_col = st.columns(2)

        with risk_col:
            risk_icon = {
                "low": "🟢",
                "moderate": "🟡",
                "high": "🟠",
                "very high": "🔴"
            }.get(risk_level.lower(), "ℹ️")
            card_class = "healthy" if risk_level.lower() == "low" else "diseased"

            st.markdown(
                f'''<div class="result-card {card_class}">
                    <div class="result-badge">Weather Risk</div>
                    <div class="result-label">{risk_icon} {risk_level.upper()}</div>
                    <div class="result-confidence">Risk score: {risk_score}</div>
                </div>''',
                unsafe_allow_html=True
            )

            if risks:
                st.markdown("**⚠️ Risk factors**")
                for risk in risks:
                    st.write(f"• {risk}")
            else:
                st.success("No significant weather-related plant risk detected.")

        with advice_col:
            st.markdown(
                '<div class="section-title" style="font-size:17px;">🌱 Plant-care recommendations</div>',
                unsafe_allow_html=True
            )
            if recommendations:
                for recommendation in recommendations:
                    st.info(recommendation)
            else:
                st.write(
                    "Weather conditions are currently relatively favorable. "
                    "Continue normal plant care and monitor the plant regularly."
                )

        st.caption(
            f'📍 {current_weather.get("location", weather_location)} · '
            f'Updated: {current_weather.get("time", "current")}'
        )
    else:
        st.markdown(
            """
            <div class="gcard" style="text-align:center; padding:40px 28px;">
                <div style="font-size:36px; margin-bottom:10px;">🌦️</div>
                <div class="section-sub" style="margin-bottom:0;">Enter a location and check weather risk to see live conditions here.</div>
            </div>
            """,
            unsafe_allow_html=True
        )


# ==================================================
# PAGE: GUIDELINES
# ==================================================

def render_guidelines():
    history = get_prediction_history()
    topbar("📋", "Guidelines", "How the tool works, and how to get the most reliable results.", history)

    col1, col2 = st.columns(2)

    with col1:
        with st.container(key="card-guide-how"):
            st.markdown('<div class="section-title" style="font-size:18px;">🧭 How it works</div>', unsafe_allow_html=True)
            steps = [
                "Upload or capture a leaf photo",
                "MobileNetV2 analyzes tissue patterns",
                "Review the diagnosis & Grad-CAM map",
                "Track results in your history log",
            ]
            html = ""
            for i, step in enumerate(steps, start=1):
                html += f"""
                <div class="guide-step">
                    <div class="num">{i}</div>
                    <div class="txt">{step}</div>
                </div>
                """
            st.markdown(html, unsafe_allow_html=True)

    with col2:
        with st.container(key="card-guide-photo"):
            st.markdown('<div class="section-title" style="font-size:18px;">📸 Getting a reliable photo</div>', unsafe_allow_html=True)
            tips = [
                "Photograph a single leaf against a plain background",
                "Use even, natural light — avoid harsh shadows or glare",
                "Fill the frame with the leaf and keep it in focus",
                "Capture both healthy and suspect areas of the leaf",
            ]
            html = ""
            for i, tip in enumerate(tips, start=1):
                html += f"""
                <div class="guide-step">
                    <div class="num">{i}</div>
                    <div class="txt">{tip}</div>
                </div>
                """
            st.markdown(html, unsafe_allow_html=True)

    st.write("")

    with st.container(key="card-guide-legend"):
        st.markdown('<div class="section-title" style="font-size:18px;">🎨 Reading the Grad-CAM heatmap</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-sub">The Attention Guide on the Grad-CAM Heatmap page uses this color scale.</div>',
            unsafe_allow_html=True
        )
        legend_items = [
            ("#E63946", "Red", "Very high attention"),
            ("#F4A261", "Orange / Yellow", "High attention"),
            ("#52B788", "Green", "Moderate attention"),
            ("#457B9D", "Blue", "Low attention"),
        ]
        legend_html = ""
        for color, label, desc in legend_items:
            legend_html += f"""
            <div class="legend-row">
                <div class="legend-dot" style="background-color:{color};"></div>
                <div><b>{label}</b>&nbsp;&nbsp;— {desc}</div>
            </div>
            """
        st.markdown(legend_html, unsafe_allow_html=True)

    st.write("")
    st.info(
        "🩺 This tool provides agronomic guidance only. High Grad-CAM attention does not "
        "necessarily mean a region is diseased — for treatment decisions, consult a "
        "qualified plant health expert."
    )


# ==================================================
# PAGE: SETTINGS
# ==================================================

def render_settings():
    history = get_prediction_history()
    topbar("⚙️", "Settings", "Appearance and data preferences for this session.", history)

    col1, col2 = st.columns(2)

    with col1:
        with st.container(key="card-settings-appearance"):
            st.markdown('<div class="section-title" style="font-size:18px;">🎨 Appearance</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="section-sub">Switch between a bright, clinical light theme and a focused dark theme.</div>',
                unsafe_allow_html=True
            )
            def _sync_dark_mode():
                st.session_state["dark_mode_pref"] = st.session_state["dark_toggle"]

            st.toggle(
                "🌙 Dark mode",
                value=st.session_state["dark_mode_pref"],
                key="dark_toggle",
                on_change=_sync_dark_mode,
            )
            st.caption(f"Currently using the **{st.session_state['theme']}** theme.")

    with col2:
        with st.container(key="card-settings-about"):
            st.markdown('<div class="section-title" style="font-size:18px;">ℹ️ About this tool</div>', unsafe_allow_html=True)
            st.markdown(
                """<div style="font-size:13.5px; line-height:1.7;">
                Powered by MobileNetV2 &amp; Grad-CAM.<br>
                Predictions include a confidence score and a visual
                explanation of the regions the model focused on.<br><br>
                For agronomic guidance only — consult an expert for
                treatment decisions.
                </div>""",
                unsafe_allow_html=True
            )

    st.write("")

    with st.container(key="card-settings-history"):
        st.markdown('<div class="section-title" style="font-size:18px;">📜 Prediction History</div>', unsafe_allow_html=True)

        if len(history) == 0:
            st.info("No predictions have been recorded yet.")
        else:
            history_sorted = history[::-1]
            st.markdown('<div class="section-sub">🧾 Previous analyses, most recent first.</div>', unsafe_allow_html=True)

            rows_html = ""
            for prediction in history_sorted:
                hist_result = prediction["result"]
                hist_confidence = prediction["confidence"]
                hist_time = prediction["time"]

                if hist_result.lower() == "healthy":
                    icon = "🟢"
                    pill_class = "healthy"
                else:
                    icon = "🔴"
                    pill_class = "diseased"

                rows_html += f"""
                <div class="hist-row">
                    <div class="hist-left">
                        <span style="font-size:18px;">{icon}</span>
                        <span class="hist-pill {pill_class}">{hist_result}</span>
                        <span class="hist-time">🕐 {hist_time}</span>
                    </div>
                    <div class="hist-conf">🎯 {hist_confidence:.2f}%</div>
                </div>
                """

            st.markdown(rows_html, unsafe_allow_html=True)

            st.write("")
            if st.button("🗑️ Clear Prediction History"):
                clear_prediction_history()
                st.success("Prediction history cleared.")
                st.rerun()


# ==================================================
# ROUTER
# ==================================================

if active_page == "🏠 Home":
    render_home()
elif active_page == "🔬 Disease Detection":
    render_disease_detection()
elif active_page == "📈 Confidence Graph":
    render_confidence_graph()
elif active_page == "🔥 Grad-CAM Heatmap":
    render_gradcam_heatmap()
elif active_page == "🌦️ Weather & Risk":
    render_weather_risk()
elif active_page == "📋 Guidelines":
    render_guidelines()
elif active_page == "⚙️ Settings":
    render_settings()