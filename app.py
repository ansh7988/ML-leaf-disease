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


# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Leaf Health AI | Dashboard",
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
    "pred_healthy_probability": None,
    "pred_diseased_probability": None,
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

    --radius-lg: 22px;
    --radius-md: 16px;
    --radius-sm: 10px;
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

/* Remove Streamlit's own top header entirely (Deploy button, toolbar,
   status spinner, colored decoration line). Previously this header was
   kept around only because it also housed the sidebar collapse/expand
   control — but the sidebar is now forced permanently open below, so
   that control is gone too and the header has nothing left to do.
   Leaving it visible was actually the cause of the black "Deploy" box
   overlapping the topbar stats. The bare "header" tag selector (in
   addition to the data-testid ones) makes this resilient to Streamlit
   renaming its internal test-ids across versions. */
header,
header[data-testid="stHeader"],
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
/* Keep the navigation sidebar permanently expanded and on-screen.
   Streamlit lets the sidebar be collapsed (via its own toggle, a stray
   click, or a narrow window) and — depending on the Streamlit version —
   the button to bring it back can end up mis-positioned or hidden. Since
   the sidebar is this app's only navigation, we simply stop it from ever
   sliding away, so Home/Disease Detection/Confidence Graph/etc. are
   always reachable regardless of that collapsed state or button quirk. */
section[data-testid="stSidebar"] {
    transform: none !important;
    visibility: visible !important;
    position: relative !important;
    margin-left: 0px !important;
    min-width: 21rem !important;
    max-width: 21rem !important;
    width: 21rem !important;
}
section[data-testid="stSidebar"] > div:first-child {
    transform: none !important;
    margin-left: 0px !important;
    visibility: visible !important;
    width: 21rem !important;
}
/* The collapse/expand toggle no longer serves a purpose since the
   sidebar can't be collapsed anymore — hide it instead of leaving a
   dead button on screen. */
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"],
button[data-testid="baseButton-headerNoPadding"] {
    display: none !important;
}

.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background:
        radial-gradient(circle at 8% 0%, rgba(82,183,136,0.10) 0%, transparent 45%),
        radial-gradient(circle at 95% 8%, rgba(45,106,79,0.10) 0%, transparent 40%),
        var(--bg);
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
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 10px 14px !important;
    transition: background 0.15s ease, border-color 0.15s ease;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background: rgba(255,255,255,0.10);
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
    background: linear-gradient(135deg, var(--leaf), var(--sprout));
    border-color: transparent;
    box-shadow: 0 4px 14px rgba(0,0,0,0.25);
}
section[data-testid="stSidebar"] div[role="radiogroup"] label p {
    font-size: 14px !important;
    font-weight: 600 !important;
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
    background: var(--topbar-bg);
    border: 1px solid var(--topbar-border);
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
    color: var(--heading);
    display: flex;
    align-items: center;
    gap: 10px;
}
.topbar-sub {
    font-size: 13px;
    color: var(--muted);
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
    color: var(--heading);
}
.topbar-stat .lab {
    font-size: 10.5px;
    color: var(--muted);
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
    background: var(--surface);
    border-radius: var(--radius-md);
    padding: 26px 28px;
    box-shadow: var(--shadow-sm);
    border: 1px solid var(--border);
    height: 100%;
}

/* Real Streamlit containers (st.container(key=...)) sharing the "card-"
   key prefix all get the same card treatment applied to the actual
   wrapping element Streamlit renders. */
div[data-testid="stVerticalBlockBorderWrapper"][class*="st-key-card-"] {
    background: var(--surface);
    border-radius: var(--radius-md);
    padding: 26px 28px;
    box-shadow: var(--shadow-sm);
    border: 1px solid var(--border);
    height: 100%;
}

/* Quick-action / nav cards get a hover lift */
div[data-testid="stVerticalBlockBorderWrapper"][class*="st-key-qa-"] {
    background: var(--surface);
    border-radius: var(--radius-md);
    padding: 22px 22px;
    box-shadow: var(--shadow-sm);
    border: 1px solid var(--border);
    height: 100%;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
div[data-testid="stVerticalBlockBorderWrapper"][class*="st-key-qa-"]:hover {
    transform: translateY(-3px);
    box-shadow: var(--shadow-md);
}

.section-title {
    font-size: 22px;
    font-weight: 700;
    color: var(--heading);
    margin-bottom: 4px;
}
.section-sub {
    font-size: 14px;
    color: var(--muted);
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
    background: var(--chip-bg);
    border-radius: var(--radius-sm);
    padding: 14px 16px;
    text-align: center;
}
.stat-chip .num { font-size: 20px; font-weight: 700; color: var(--chip-text); font-family:'Poppins',sans-serif; }
.stat-chip .lab { font-size: 12px; color: var(--muted); margin-top: 2px; }

.icon-badge {
    width: 44px; height: 44px;
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 20px;
    background: var(--chip-bg);
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
div[data-testid="stAlert"] p {
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
    background: var(--hist-row-bg);
    border: 1px solid var(--border);
    margin-bottom: 10px;
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
.hist-pill.healthy { background: var(--pill-healthy-bg); color: var(--pill-healthy-text); }
.hist-pill.diseased { background: var(--pill-diseased-bg); color: var(--pill-diseased-text); }
.hist-time { color: var(--muted); font-size: 13px; }
.hist-conf { font-weight: 700; color: var(--heading); font-family:'Poppins',sans-serif; }

/* ---------- GUIDELINE STEPS ---------- */

.guide-step {
    display: flex;
    gap: 14px;
    padding: 14px 0;
    border-bottom: 1px dashed var(--border);
}
.guide-step:last-child { border-bottom: none; }
.guide-step .num {
    flex-shrink: 0;
    width: 30px; height: 30px;
    border-radius: 50%;
    background: var(--chip-bg);
    color: var(--heading);
    font-weight: 700;
    font-family: 'Poppins', sans-serif;
    display: flex; align-items: center; justify-content: center;
    font-size: 13px;
}
.guide-step .txt b { color: var(--heading); }

/* ---------- MISC ---------- */

hr { margin: 1.6rem 0; border-color: var(--border); }

.stRadio > div { gap: 8px; }
.stRadio label {
    background: var(--surface);
    border: 1px solid var(--border);
    padding: 8px 16px;
    border-radius: 999px;
}

.stButton > button {
    background: linear-gradient(135deg, var(--leaf), var(--sprout));
    color: white;
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
    color: white;
}
.stButton > button:focus-visible {
    outline: 2px solid var(--sprout);
    outline-offset: 2px;
}

div[data-testid="stFileUploaderDropzone"] {
    background: var(--uploader-bg);
    border: 2px dashed var(--uploader-border);
    border-radius: var(--radius-md);
}

div[data-testid="stTextInput"] input {
    background: var(--input-bg);
    color: var(--text);
    border: 1px solid var(--border);
}

img { border-radius: var(--radius-sm); }

/* Chart / photo panels are framed directly on the image element itself
   (never on a wrapping container) — this is what actually renders the
   card look for st.image()/st.pyplot() output, deliberately avoiding
   st.container(key=...) around them, which is prone to leaving a stray
   empty bordered box above the real content. */
div[data-testid="stImage"] img {
    border-radius: var(--radius-md);
    box-shadow: var(--shadow-sm);
    border: 1px solid var(--border);
    background: var(--surface);
}

.stProgress > div > div > div > div {
    background: linear-gradient(90deg, var(--leaf), var(--sprout));
}

/* Toggle label readability in Settings */
[data-testid="stWidgetLabel"] p { color: var(--text) !important; }

</style>
"""

st.markdown(_root_vars + _static_css, unsafe_allow_html=True)

# JS-based fallback for hiding Streamlit's own header/Deploy button.
# The CSS rule above already targets it, but some Streamlit builds use a
# data-testid the CSS misses, or re-apply an inline style after our
# stylesheet loads. This walks the real page DOM (via window.parent,
# since components.html renders inside an iframe) and force-hides it
# directly, then keeps re-checking so it can't reappear after a rerun.
# It only ever touches Streamlit's own chrome — never the app's content.
components.html(
    """
    <script>
    (function () {
        function hideStreamlitChrome() {
            try {
                const doc = window.parent.document;
                const selectors = [
                    'header', '[data-testid="stHeader"]', '[data-testid="stToolbar"]',
                    '[data-testid="stDecoration"]', '[data-testid="stStatusWidget"]',
                    '[data-testid="stAppDeployButton"]', '[data-testid="stToolbarActions"]',
                    '[data-testid="stAppToolbar"]'
                ];
                doc.querySelectorAll(selectors.join(',')).forEach(function (el) {
                    el.style.setProperty('display', 'none', 'important');
                    el.style.setProperty('visibility', 'hidden', 'important');
                    el.style.setProperty('pointer-events', 'none', 'important');
                    el.style.setProperty('height', '0px', 'important');
                });
                doc.querySelectorAll('button, span, div').forEach(function (el) {
                    if (el.children.length === 0 && el.textContent.trim() === 'Deploy') {
                        var target = el.closest('header') || el;
                        target.style.setProperty('display', 'none', 'important');
                    }
                });
            } catch (e) { /* not ready yet — ignore, next tick will retry */ }
        }
        hideStreamlitChrome();
        try {
            new MutationObserver(hideStreamlitChrome)
                .observe(window.parent.document.body, { childList: true, subtree: true });
        } catch (e) {}
        setInterval(hideStreamlitChrome, 400);
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
                {stats_html}
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
            <div class="brand-name">Leaf Health AI</div>
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

        is_healthy = st.session_state["pred_result"].lower() == "healthy"
        card_class = "healthy" if is_healthy else "diseased"
        result_icon = "🟢" if is_healthy else "🔴"
        status_text = "No disease detected" if is_healthy else "Attention recommended"

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
                result, confidence, healthy_probability, diseased_probability = predict_leaf(image_path)

            with st.spinner("Generating AI visual.."):
                gradcam_image = make_gradcam(image_path)

            st.session_state["leaf_image_bytes"] = image_bytes
            st.session_state["leaf_image_name"] = getattr(image_file, "name", "leaf.jpg")
            st.session_state["leaf_image_hash"] = image_hash
            st.session_state["pred_result"] = result
            st.session_state["pred_confidence"] = confidence
            st.session_state["pred_healthy_probability"] = healthy_probability
            st.session_state["pred_diseased_probability"] = diseased_probability
            st.session_state["gradcam_image"] = gradcam_image
        else:
            result = st.session_state["pred_result"]
            confidence = st.session_state["pred_confidence"]

        st.markdown("<hr>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="section-title" style="font-size:18px;">🍃 Leaf Image</div>', unsafe_allow_html=True)
            st.image(io.BytesIO(st.session_state["leaf_image_bytes"]), use_container_width=True)

        with col2:
            is_healthy = result.lower() == "healthy"
            result_icon = "🟢" if is_healthy else "🔴"
            card_class = "healthy" if is_healthy else "diseased"
            status_text = "No disease detected" if is_healthy else "Attention recommended"

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
    with graph1:
        st.markdown('<div class="section-title" style="font-size:17px;">🧠 Prediction Probability</div>', unsafe_allow_html=True)

        labels = ["Healthy", "Diseased"]

        if result.lower() == "healthy":
            probabilities = [confidence * 100, (1 - confidence) * 100]
        else:
            probabilities = [(1 - confidence) * 100, confidence * 100]

        fig1, ax1 = plt.subplots(figsize=(6.4, 3.8))
        bar_colors = [SPROUT, SUN]
        ax1.bar(labels, probabilities, color=bar_colors, width=0.5, zorder=3)

        for spine in ["top", "right", "left"]:
            ax1.spines[spine].set_visible(False)

        ax1.set_ylim(0, 100)
        ax1.set_ylabel("Probability (%)")
        ax1.grid(axis="y", alpha=0.25, zorder=0)
        ax1.set_axisbelow(True)

        for i, value in enumerate(probabilities):
            ax1.text(i, value + 3, f"{value:.2f}%", ha="center", fontweight="bold", color=GRAPH_ACCENT)

        plt.tight_layout()
        st.pyplot(fig1, transparent=True)
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