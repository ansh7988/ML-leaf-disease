import streamlit as st
import tempfile
import os
import matplotlib.pyplot as plt

from src.predict import predict_leaf
from src.prediction_history import get_prediction_history, clear_prediction_history


# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Leaf Health AI",
    page_icon="🌿",
    layout="wide"
)


# ==================================================
# CUSTOM CSS
# ==================================================

st.markdown("""
<style>

.main {
    background-color: #f7faf8;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #6b7280;
    margin-bottom: 30px;
}

.result-card {
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    background-color: #ffffff;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
}

.result {
    font-size: 32px;
    font-weight: 700;
}

.confidence {
    font-size: 22px;
    margin-top: 10px;
}

</style>
""", unsafe_allow_html=True)


# ==================================================
# HEADER
# ==================================================

st.markdown(
    '<div class="title">🌿 Leaf Health AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered leaf health analysis using MobileNetV2'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ==================================================
# INPUT METHOD
# ==================================================

st.subheader("🔍 Choose Analysis Method")

method = st.radio(
    "How would you like to provide the leaf image?",
    ["📁 Upload Image", "📷 Use Camera"],
    horizontal=True
)

image_file = None


# ==================================================
# UPLOAD IMAGE
# ==================================================

if method == "📁 Upload Image":

    image_file = st.file_uploader(
        "Choose a leaf image",
        type=["jpg", "jpeg", "png", "webp"]
    )


# ==================================================
# CAMERA
# ==================================================

else:

    image_file = st.camera_input(
        "Take a picture of the leaf"
    )


# ==================================================
# ANALYSIS
# ==================================================

if image_file is not None:

    st.divider()

    col1, col2 = st.columns(2)


    # ==================================================
    # IMAGE PREVIEW
    # ==================================================

    with col1:

        st.subheader("🍃 Leaf Image")

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

        st.subheader("🔬 Analysis")

        with st.spinner(
            "Analyzing leaf..."
        ):

            result, confidence, healthy_probability, diseased_probability = predict_leaf(
                image_path
            )


        # ----------------------------------------------
        # RESULT ICON
        # ----------------------------------------------

        if result.lower() == "healthy":

            result_icon = "🟢"

        else:

            result_icon = "🔴"


        # ----------------------------------------------
        # RESULT CARD
        # ----------------------------------------------

        st.markdown(
            f"""
            <div class="result-card">

                <div class="result">
                    {result_icon} {result.upper()}
                </div>

                <div class="confidence">
                    Confidence: {confidence * 100:.2f}%
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


        # ----------------------------------------------
        # CONFIDENCE PROGRESS
        # ----------------------------------------------

        st.progress(
            confidence
        )


    # ==================================================
    # DELETE TEMPORARY IMAGE
    # ==================================================

    try:

        os.remove(image_path)

    except Exception:

        pass


    # ==================================================
    # AI ANALYSIS RESULTS
    # ==================================================

    st.divider()

    st.subheader("📊 AI Analysis Results")


    # ==================================================
    # CREATE TWO GRAPH COLUMNS
    # ==================================================

    graph1, graph2 = st.columns(2)


    # ==================================================
    # GRAPH 1 — PREDICTION PROBABILITY
    # ==================================================

    with graph1:

        st.markdown(
            "### 🧠 Prediction Probability"
        )


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
            figsize=(7, 4)
        )


        ax1.bar(
            labels,
            probabilities
        )


        ax1.set_ylim(
            0,
            100
        )


        ax1.set_ylabel(
            "Probability (%)"
        )


        ax1.set_title(
            "Model Prediction Probability"
        )


        # Add percentage above bars

        for i, value in enumerate(
            probabilities
        ):

            ax1.text(
                i,
                value + 2,
                f"{value:.2f}%",
                ha="center",
                fontweight="bold"
            )


        plt.tight_layout()


        st.pyplot(
            fig1
        )


        plt.close(
            fig1
        )


    # ==================================================
    # GRAPH 2 — CONFIDENCE LINE GRAPH
    # ==================================================

    with graph2:

        st.markdown(
            "### 🎯 Model Confidence"
        )


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
            figsize=(7, 4)
        )


        # Line

        ax2.plot(
            x,
            y,
            marker="o",
            linewidth=3,
            markersize=7
        )


        # Highlight final point

        ax2.scatter(
            x[-1],
            y[-1],
            s=120,
            zorder=5
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
            fontweight="bold"
        )


        # ----------------------------------------------
        # Graph settings
        # ----------------------------------------------

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


        ax2.set_title(
            f"Confidence in {result} Prediction"
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
            alpha=0.25
        )


        plt.xticks(
            rotation=20
        )


        plt.tight_layout()


        st.pyplot(
            fig2
        )


        plt.close(
            fig2
        )

        # ==================================================
        # PREDICTION HISTORY
        # ==================================================

        st.divider()

        st.subheader("📜 Prediction History")

        history = get_prediction_history()


        if len(history) == 0:

            st.info(
                "No predictions have been recorded yet."
            )

        else:

            # Show newest prediction first
            history = history[::-1]


            # --------------------------------------------------
            # HISTORY TABLE
            # --------------------------------------------------

            st.markdown(
                "### 🧾 Previous Analyses"
            )


            for i, prediction in enumerate(history):

                result = prediction["result"]
                confidence = prediction["confidence"]
                time = prediction["time"]


                if result.lower() == "healthy":

                    icon = "🟢"

                else:

                    icon = "🔴"


                col1, col2, col3, col4 = st.columns(
                    [2, 2, 2, 2]
                )


                with col1:

                    st.write(
                        f"**{icon} {result}**"
                    )




                with col2:

                    st.write(
                        f"🕐 {time}"
                    )


                with col3:

                    st.write(
                        f"🎯 {confidence:.2f}%"
                    )


                if i < len(history) - 1:

                    st.divider()


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