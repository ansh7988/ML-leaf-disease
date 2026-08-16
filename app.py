import streamlit as st
import tempfile
import os
import matplotlib.pyplot as plt
from gradcam import make_gradcam
from src.predict import predict_leaf
from src.prediction_history import get_prediction_history, clear_prediction_history


# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Leaf Health AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================================================
# GLOBAL STYLE TOKENS + CSS
# ==================================================
# Design system:
#   Forest   #1B4332   (deep canopy, headings)
#   Leaf     #2D6A4F   (primary brand green)
#   Sprout   #52B788   (accent / highlights)
#   Sage     #D8F3DC   (soft backgrounds, chips)
#   Sun      #F4A261   (warning / diseased accent)
#   Canvas   #F7FAF8   (page background)
#   Ink      #1F2937   (body text)
# Type: "Poppins" for display headings, "Inter" for body/data.

st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@500;600;700;800&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">

<style>

:root {
    --forest: #1B4332;
    --leaf: #2D6A4F;
    --sprout: #52B788;
    --sprout-soft: #95D5B2;
    --sage: #D8F3DC;
    --sun: #F4A261;
    --danger: #E76F51;
    --canvas: #F7FAF8;
    --card: #FFFFFF;
    --ink: #1F2937;
    --muted: #6B7280;
    --border: rgba(27, 67, 50, 0.08);
    --shadow-sm: 0 2px 10px rgba(27, 67, 50, 0.06);
    --shadow-md: 0 10px 30px rgba(27, 67, 50, 0.10);
    --radius-lg: 22px;
    --radius-md: 16px;
    --radius-sm: 10px;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: var(--ink);
}

h1, h2, h3, h4, .app-font-display {
    font-family: 'Poppins', sans-serif !important;
}

.stApp {
    background:
        radial-gradient(circle at 8% 0%, rgba(82,183,136,0.10) 0%, transparent 45%),
        radial-gradient(circle at 95% 10%, rgba(45,106,79,0.08) 0%, transparent 40%),
        var(--canvas);
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--forest) 0%, var(--leaf) 100%);
}
section[data-testid="stSidebar"] * {
    color: #EAF6EF !important;
}
section[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.18);
}

#MainMenu, footer {visibility: hidden;}

.block-container {
    padding-top: 1.6rem;
    padding-bottom: 3rem;
}

/* ---------- HERO ---------- */

.hero {
    position: relative;
    overflow: hidden;
    padding: 46px 44px;
    border-radius: var(--radius-lg);
    background: linear-gradient(120deg, var(--forest) 0%, var(--leaf) 55%, var(--sprout) 130%);
    box-shadow: var(--shadow-md);
    margin-bottom: 28px;
}

.hero::before {
    content: "";
    position: absolute;
    top: -60px;
    right: -60px;
    width: 260px;
    height: 260px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(255,255,255,0.16) 0%, transparent 70%);
}

.hero::after {
    content: "";
    position: absolute;
    bottom: -90px;
    left: 20%;
    width: 320px;
    height: 320px;
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
    font-size: 40px;
    font-weight: 800;
    color: #FFFFFF;
    line-height: 1.15;
    margin-bottom: 10px;
}

.hero-subtitle {
    font-size: 16.5px;
    color: rgba(255,255,255,0.86);
    max-width: 620px;
    line-height: 1.55;
}

/* ---------- GENERIC CARD ---------- */

.gcard {
    background: var(--card);
    border-radius: var(--radius-md);
    padding: 26px 28px;
    box-shadow: var(--shadow-sm);
    border: 1px solid var(--border);
    height: 100%;
}

/* Real Streamlit containers used as cards (st.container(key=...)) get this
   look applied to the actual wrapping element Streamlit renders, instead of
   a hand-written <div> that gets auto-closed inside its own st.markdown()
   call (which was causing the empty white bars / floating headings). */
div[data-testid="stVerticalBlockBorderWrapper"].st-key-leaf-image-card,
div[data-testid="stVerticalBlockBorderWrapper"].st-key-pred-prob-card,
div[data-testid="stVerticalBlockBorderWrapper"].st-key-model-conf-card,
div[data-testid="stVerticalBlockBorderWrapper"].st-key-orig-image-card,
div[data-testid="stVerticalBlockBorderWrapper"].st-key-gradcam-card,
div[data-testid="stVerticalBlockBorderWrapper"].st-key-attention-card {
    background: var(--card);
    border-radius: var(--radius-md);
    padding: 26px 28px;
    box-shadow: var(--shadow-sm);
    border: 1px solid var(--border);
    height: 100%;
}

.section-title {
    font-size: 22px;
    font-weight: 700;
    color: var(--forest);
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

.result-card.healthy {
    background: linear-gradient(135deg, #2D6A4F 0%, #52B788 100%);
}

.result-card.diseased {
    background: linear-gradient(135deg, #9D2B2B 0%, #E76F51 100%);
}

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

.result-confidence {
    font-size: 16px;
    opacity: 0.92;
    font-weight: 500;
}

/* ---------- STAT / MINI CARDS ---------- */

.stat-chip {
    background: var(--sage);
    border-radius: var(--radius-sm);
    padding: 14px 16px;
    text-align: center;
}
.stat-chip .num { font-size: 20px; font-weight: 700; color: var(--forest); font-family:'Poppins',sans-serif;}
.stat-chip .lab { font-size: 12px; color: var(--muted); margin-top: 2px;}

/* ---------- LEGEND ---------- */

.legend-row {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 9px 0;
    color: var(--ink) !important;
}
.legend-row b {
    color: var(--forest) !important;
}
.legend-dot {
    width: 16px;
    height: 16px;
    border-radius: 5px;
    flex-shrink: 0;
    box-shadow: 0 0 0 3px rgba(0,0,0,0.03);
}

/* ---------- FORCE READABLE TEXT ON LIGHT SURFACES ---------- */
/* Streamlit's dark-theme default text color is near-white, which is */
/* invisible on our light cards/canvas. Force it explicitly wherever */
/* we are not deliberately painting text white (hero / sidebar / result-card). */

[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span:not(.legend-dot),
[data-testid="stCaptionContainer"] p,
.stRadio label p,
div[data-testid="stFileUploaderDropzone"] * ,
div[data-testid="stAlert"] p {
    color: var(--ink) !important;
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
    background: #FBFDFB;
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
.hist-pill.healthy { background: var(--sage); color: var(--leaf); }
.hist-pill.diseased { background: #FBE3DA; color: #C1502E; }
.hist-time { color: var(--muted); font-size: 13px; }
.hist-conf { font-weight: 700; color: var(--forest); font-family:'Poppins',sans-serif; }

/* ---------- MISC ---------- */

hr { margin: 1.6rem 0; border-color: var(--border); }

.stRadio > div { gap: 8px; }
.stRadio label {
    background: var(--card);
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

div[data-testid="stFileUploaderDropzone"] {
    background: var(--sage);
    border: 2px dashed var(--sprout-soft);
    border-radius: var(--radius-md);
}

img { border-radius: var(--radius-sm); }

.stProgress > div > div > div > div {
    background: linear-gradient(90deg, var(--leaf), var(--sprout));
}

</style>
""", unsafe_allow_html=True)


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.markdown(
        """
        <div style="text-align:center; padding: 6px 0 18px 0;">
            <div style="font-size:44px;">🌿</div>
            <div style="font-family:'Poppins',sans-serif; font-weight:700; font-size:20px; color:#FFFFFF;">
                Leaf Health AI
            </div>
            <div style="font-size:12.5px; opacity:0.85; margin-top:2px;">
                Precision plant diagnostics
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("##### 🧭 How it works")
    st.markdown(
        """
        <div style="font-size:13.5px; line-height:1.7; opacity:0.92;">
        1&nbsp;&nbsp;Upload or capture a leaf photo<br>
        2&nbsp;&nbsp;MobileNetV2 analyzes tissue patterns<br>
        3&nbsp;&nbsp;Review the diagnosis &amp; Grad-CAM map<br>
        4&nbsp;&nbsp;Track results in your history log
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    history_preview = get_prediction_history()
    total_scans = len(history_preview)
    healthy_scans = sum(1 for h in history_preview if h["result"].lower() == "healthy")

    st.markdown("##### 📊 Session stats")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f"""<div class="stat-chip" style="background:rgba(255,255,255,0.12);">
            <div class="num" style="color:#FFFFFF;">{total_scans}</div>
            <div class="lab" style="color:rgba(255,255,255,0.75);">Total Scans</div>
            </div>""",
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            f"""<div class="stat-chip" style="background:rgba(255,255,255,0.12);">
            <div class="num" style="color:#FFFFFF;">{healthy_scans}</div>
            <div class="lab" style="color:rgba(255,255,255,0.75);">Healthy</div>
            </div>""",
            unsafe_allow_html=True
        )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        """<div style="font-size:11.5px; opacity:0.7; line-height:1.5;">
        Powered by MobileNetV2 &amp; Grad-CAM.<br>
        For agronomic guidance only — consult an
        expert for treatment decisions.
        </div>""",
        unsafe_allow_html=True
    )


# ==================================================
# HERO
# ==================================================

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


# ==================================================
# INPUT METHOD
# ==================================================

st.markdown('<div class="section-title">🔍 Choose Analysis Method</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Provide a clear, well-lit photo of a single leaf for the most accurate result.</div>', unsafe_allow_html=True)

method = st.radio(
    "How would you like to provide the leaf image?",
    ["📁 Upload Image", "📷 Use Camera"],
    horizontal=True,
    label_visibility="collapsed"
)

image_file = None


# ==================================================
# UPLOAD IMAGE
# ==================================================

if method == "📁 Upload Image":

    image_file = st.file_uploader(
        "Choose a leaf image",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed"
    )


# ==================================================
# CAMERA
# ==================================================

else:

    image_file = st.camera_input(
        "Take a picture of the leaf",
        label_visibility="collapsed"
    )


# ==================================================
# ANALYSIS
# ==================================================

if image_file is not None:

    st.markdown("<hr>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)


    # ==================================================
    # IMAGE PREVIEW
    # ==================================================

    with col1:

        with st.container(key="leaf-image-card"):

            st.markdown('<div class="section-title" style="font-size:18px;">🍃 Leaf Image</div>', unsafe_allow_html=True)

            st.image(
                image_file,
                use_container_width=True
            )


    # ==================================================
    # SAVE TEMPORARY IMAGE
    # ==================================================

    suffix = ".jpg"

    if hasattr(image_file, "name"):

        extension = os.path.splitext(
            image_file.name
        )[1]

        if extension:

            suffix = extension


    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    ) as temp_file:

        temp_file.write(
            image_file.getvalue()
        )

        image_path = temp_file.name


    # ==================================================
    # AI PREDICTION
    # ==================================================

    with col2:

        with st.spinner(
            "Analyzing leaf..."
        ):

            result, confidence, healthy_probability, diseased_probability = predict_leaf(
                image_path
            )

        with st.spinner(
            "Generating AI visual.."
        ):

            gradcam_image = make_gradcam(
                image_path
            )


        # ----------------------------------------------
        # RESULT CARD
        # ----------------------------------------------

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


        # ----------------------------------------------
        # CONFIDENCE PROGRESS
        # ----------------------------------------------

        st.progress(
            confidence
        )


    # ==================================================
    # AI ANALYSIS RESULTS
    # ==================================================

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">📊 AI Analysis Results</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">A closer look at how the model arrived at its decision.</div>', unsafe_allow_html=True)

    # Shared matplotlib theme

    plt.rcParams.update({
        "font.family": "sans-serif",
        "axes.edgecolor": "#E5E9E6",
        "axes.labelcolor": "#1F2937",
        "text.color": "#1F2937",
        "xtick.color": "#6B7280",
        "ytick.color": "#6B7280",
        "figure.facecolor": "none",
        "axes.facecolor": "none",
        "savefig.facecolor": "none",
    })

    FOREST = "#1B4332"
    LEAF = "#2D6A4F"
    SPROUT = "#52B788"
    SUN = "#E76F51"


    # ==================================================
    # CREATE TWO GRAPH COLUMNS
    # ==================================================

    graph1, graph2 = st.columns(2)


    # ==================================================
    # GRAPH 1 — PREDICTION PROBABILITY
    # ==================================================

    with graph1:

        with st.container(key="pred-prob-card"):

            st.markdown('<div class="section-title" style="font-size:17px;">🧠 Prediction Probability</div>', unsafe_allow_html=True)


            labels = [
                "Healthy",
                "Diseased"
            ]


            # Calculate probabilities

            if result.lower() == "healthy":

                probabilities = [
                    confidence * 100,
                    (1 - confidence) * 100
                ]

            else:

                probabilities = [
                    (1 - confidence) * 100,
                    confidence * 100
                ]


            # Create graph

            fig1, ax1 = plt.subplots(
                figsize=(6.4, 3.8)
            )

            bar_colors = [SPROUT, SUN]

            bars = ax1.bar(
                labels,
                probabilities,
                color=bar_colors,
                width=0.5,
                zorder=3
            )

            for spine in ["top", "right", "left"]:
                ax1.spines[spine].set_visible(False)

            ax1.set_ylim(0, 100)
            ax1.set_ylabel("Probability (%)")
            ax1.grid(axis="y", alpha=0.25, zorder=0)
            ax1.set_axisbelow(True)

            # Add percentage above bars

            for i, value in enumerate(
                probabilities
            ):

                ax1.text(
                    i,
                    value + 3,
                    f"{value:.2f}%",
                    ha="center",
                    fontweight="bold",
                    color=FOREST
                )


            plt.tight_layout()


            st.pyplot(
                fig1,
                transparent=True
            )


            plt.close(
                fig1
            )


    # ==================================================
    # GRAPH 2 — CONFIDENCE LINE GRAPH
    # ==================================================

    with graph2:

        with st.container(key="model-conf-card"):

            st.markdown('<div class="section-title" style="font-size:17px;">🎯 Model Confidence</div>', unsafe_allow_html=True)


            confidence_percent = (
                confidence * 100
            )


            # ----------------------------------------------
            # Confidence points
            # ----------------------------------------------

            x = [
                1,
                2,
                3,
                4,
                5
            ]


            y = [
                confidence_percent * 0.55,
                confidence_percent * 0.68,
                confidence_percent * 0.76,
                confidence_percent * 0.90,
                confidence_percent
            ]


            # ----------------------------------------------
            # Create graph
            # ----------------------------------------------

            fig2, ax2 = plt.subplots(
                figsize=(6.4, 3.8)
            )


            # Line

            ax2.plot(
                x,
                y,
                marker="o",
                linewidth=3,
                markersize=7,
                color=LEAF,
                zorder=3
            )

            ax2.fill_between(x, y, color=SPROUT, alpha=0.15, zorder=2)


            # Highlight final point

            ax2.scatter(
                x[-1],
                y[-1],
                s=130,
                zorder=5,
                color=SUN if not is_healthy else SPROUT,
                edgecolor="white",
                linewidth=1.5
            )


            # ----------------------------------------------
            # Final percentage
            # ----------------------------------------------

            ax2.annotate(
                f"{confidence_percent:.2f}%",
                (
                    x[-1],
                    y[-1]
                ),
                xytext=(10, 10),
                textcoords="offset points",
                fontsize=12,
                fontweight="bold",
                color=FOREST
            )


            # ----------------------------------------------
            # Graph settings
            # ----------------------------------------------

            for spine in ["top", "right"]:
                ax2.spines[spine].set_visible(False)

            ax2.set_ylim(
                0,
                100
            )


            ax2.set_xlabel(
                "Analysis Progress"
            )


            ax2.set_ylabel(
                "Confidence (%)"
            )


            ax2.set_xticks(
                x
            )


            ax2.set_xticklabels(
                [
                    "Start",
                    "Processing",
                    "Analysis",
                    "Finalizing",
                    "Result"
                ]
            )


            ax2.grid(
                True,
                alpha=0.25,
                zorder=0
            )
            ax2.set_axisbelow(True)


            plt.xticks(
                rotation=20
            )


            plt.tight_layout()


            st.pyplot(
                fig2,
                transparent=True
            )


            plt.close(
                fig2
            )

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">🔥 AI Visual Explanation</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Grad-CAM highlights the areas of the leaf that contributed most to the model\'s prediction.</div>',
        unsafe_allow_html=True
    )

    gradcam_col1, gradcam_col2 = st.columns(2)

    with gradcam_col1:

        with st.container(key="orig-image-card"):

            st.markdown('<div class="section-title" style="font-size:17px;">🍃 Original Image</div>', unsafe_allow_html=True)

            st.image(
                image_file,
                use_container_width=True
            )

    with gradcam_col2:

        with st.container(key="gradcam-card"):

            st.markdown('<div class="section-title" style="font-size:17px;">🔥 Grad-CAM Heatmap</div>', unsafe_allow_html=True)

            st.image(
                gradcam_image,
                use_container_width=True
            )

    st.write("")

    with st.container(key="attention-card"):

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
    # PREDICTION HISTORY
    # ==================================================

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">📜 Prediction History</div>', unsafe_allow_html=True)

    history = get_prediction_history()


    if len(history) == 0:

        st.info(
            "No predictions have been recorded yet."
        )

    else:

        # Show newest prediction first
        history = history[::-1]

        st.markdown('<div class="section-sub">🧾 Previous analyses, most recent first.</div>', unsafe_allow_html=True)

        rows_html = ""

        for prediction in history:

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

        # --------------------------------------------------
        # CLEAR HISTORY
        # --------------------------------------------------

        st.write("")

        if st.button(
            "🗑️ Clear Prediction History"
        ):

            clear_prediction_history()

            st.success(
                "Prediction history cleared."
            )

            st.rerun()