import streamlit as st
from src.database import login_user , create_user, get_user_predictions
import re
def is_valid_email(email):
    pattern = r"^[a-zA-Z0-9._%+-]+@gmail\.com$"
    return re.match(pattern, email) is not None
def show_auth():
    _static_css = """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@500;600;700;800&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">

    <style>
    /* Reset and fonts */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #1F2937;
    }

    h1, h2, h3, h4, .brand-title {
        font-family: 'Poppins', sans-serif !important;
    }

    #MainMenu, footer {
        visibility: hidden;
        display: none !important;
    }

    /* Eliminate any top black bar, header, or Streamlit chrome */
    header,
    header[data-testid="stHeader"],
    [data-testid="stHeader"],
    div[data-testid="stToolbar"],
    div[data-testid="stDecoration"],
    div[data-testid="stStatusWidget"],
    [data-testid="stAppDeployButton"],
    [data-testid="stToolbarActions"],
    [data-testid="stAppToolbar"] {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
        max-height: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
        background: transparent !important;
        border: none !important;
    }

    /* Atmospheric natural background */
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stMainBlockContainer"],
    .main {
        background:
            radial-gradient(circle at 10% 12%, rgba(82, 183, 136, 0.14) 0%, transparent 48%),
            radial-gradient(circle at 90% 88%, rgba(45, 106, 79, 0.10) 0%, transparent 52%),
            #F7FAF8 !important;
        min-height: 100vh !important;
    }

    /* Page container constraints */
    .block-container {
        padding-top: 2.2rem !important;
        padding-bottom: 2.2rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 1020px !important;
        margin: 0 auto !important;
    }

    /* Left Brand Panel */
    .brand-hero-panel {
        padding: 24px 20px 24px 8px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        height: 100%;
    }

    .brand-eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        font-size: 11.5px;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #2D6A4F;
        background: #D8F3DC;
        padding: 5px 14px;
        border-radius: 999px;
        margin-bottom: 14px;
        align-self: flex-start;
        border: 1px solid rgba(45, 106, 79, 0.12);
    }

    .brand-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: #2D6A4F;
    }

    .brand-main-title {
        font-family: 'Poppins', sans-serif !important;
        font-size: 34px !important;
        font-weight: 800 !important;
        color: #1B4332 !important;
        line-height: 1.16 !important;
        margin: 0 0 8px 0 !important;
        letter-spacing: -0.02em !important;
    }

    .brand-lead-sub {
        font-size: 15px !important;
        font-weight: 600 !important;
        color: #2D6A4F !important;
        margin: 0 0 8px 0 !important;
        line-height: 1.4 !important;
    }

    .brand-desc {
        font-size: 13.5px !important;
        color: #556B5F !important;
        line-height: 1.55 !important;
        margin: 0 0 20px 0 !important;
    }

    .feature-pills-list {
        display: flex;
        flex-direction: column;
        gap: 10px;
        margin-top: 4px;
    }

    .feature-pill-item {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 10px 14px;
        border-radius: 12px;
        background: rgba(255, 255, 255, 0.85);
        border: 1px solid rgba(27, 67, 50, 0.08);
        box-shadow: 0 2px 8px rgba(27, 67, 50, 0.03);
        transition: transform 0.15s ease, background 0.15s ease;
    }

    .feature-pill-item:hover {
        background: #FFFFFF;
        transform: translateX(3px);
    }

    .feature-icon {
        font-size: 17px;
        width: 34px;
        height: 34px;
        border-radius: 9px;
        background: #E8F7EE;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
    }

    .feature-text {
        display: flex;
        flex-direction: column;
    }

    .feature-text strong {
        font-size: 13px;
        color: #1B4332;
        font-weight: 600;
    }

    .feature-text span {
        font-size: 11.5px;
        color: #647A6E;
    }

    /* Authentication Card Container */
    div[data-testid="stVerticalBlockBorderWrapper"][class*="st-key-auth_card"],
    div[data-testid="stVerticalBlock"][class*="st-key-auth_card"],
    div[class*="st-key-auth_card"] {
        background: #FFFFFF !important;
        border-radius: 20px !important;
        padding: 28px 26px !important;
        box-shadow: 0 12px 36px rgba(27, 67, 50, 0.08), 0 2px 10px rgba(27, 67, 50, 0.04) !important;
        border: 1px solid rgba(27, 67, 50, 0.08) !important;
        box-sizing: border-box !important;
        animation: cardEntrance 0.35s ease-out forwards;
    }

    @keyframes cardEntrance {
        from {
            opacity: 0;
            transform: translateY(8px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    /* Segmented Control Switcher */
    .stRadio > label {
        display: none !important;
    }

    .stRadio div[role="radiogroup"] {
        display: flex !important;
        flex-direction: row !important;
        background: #EBF3EE !important;
        border: 1px solid rgba(27, 67, 50, 0.10) !important;
        border-radius: 999px !important;
        padding: 4px !important;
        gap: 4px !important;
        margin-bottom: 20px !important;
    }

    .stRadio div[role="radiogroup"] label {
        flex: 1 1 50% !important;
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        text-align: center !important;
        background: transparent !important;
        border: none !important;
        border-radius: 999px !important;
        padding: 8px 14px !important;
        margin: 0 !important;
        cursor: pointer !important;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }

    .stRadio div[role="radiogroup"] label:hover {
        background: rgba(45, 106, 79, 0.08) !important;
    }

    .stRadio div[role="radiogroup"] label:has(input:checked) {
        background: linear-gradient(135deg, #1B4332 0%, #2D6A4F 100%) !important;
        box-shadow: 0 2px 8px rgba(27, 67, 50, 0.25) !important;
    }

    .stRadio div[role="radiogroup"] label div[data-testid="stMarkdownContainer"] p,
    .stRadio div[role="radiogroup"] label span,
    .stRadio div[role="radiogroup"] label p {
        font-size: 13.5px !important;
        font-weight: 600 !important;
        color: #4B6354 !important;
        letter-spacing: 0.01em !important;
        margin: 0 !important;
    }

    .stRadio div[role="radiogroup"] label:has(input:checked) div[data-testid="stMarkdownContainer"] p,
    .stRadio div[role="radiogroup"] label:has(input:checked) span,
    .stRadio div[role="radiogroup"] label:has(input:checked) p {
        color: #FFFFFF !important;
    }

    /* Card Header inside form */
    .auth-form-header {
        margin-bottom: 16px;
    }

    .auth-form-title {
        font-family: 'Poppins', sans-serif !important;
        font-size: 20px !important;
        font-weight: 700 !important;
        color: #1B4332 !important;
        margin: 0 0 3px 0 !important;
    }

    .auth-form-sub {
        font-size: 13px !important;
        color: #647A6E !important;
        margin: 0 !important;
    }

    /* Text Inputs */
    div[data-testid="stTextInput"] {
        margin-bottom: 14px;
    }

    div[data-testid="stTextInput"] label p {
        font-size: 13px !important;
        font-weight: 600 !important;
        color: #1B4332 !important;
        margin-bottom: 4px !important;
        letter-spacing: 0.01em !important;
    }

    div[data-testid="stTextInput"] input {
        background: #FAFCFA !important;
        color: #1F2937 !important;
        border: 1.5px solid rgba(27, 67, 50, 0.12) !important;
        border-radius: 12px !important;
        padding: 10px 14px !important;
        font-size: 14px !important;
        font-family: 'Inter', sans-serif !important;
        transition: all 0.2s ease !important;
    }

    div[data-testid="stTextInput"] input:focus {
        background: #FFFFFF !important;
        border-color: #2D6A4F !important;
        box-shadow: 0 0 0 3px rgba(45, 106, 79, 0.15) !important;
        outline: none !important;
    }

    div[data-testid="stTextInput"] input::placeholder {
        color: #8C9E94 !important;
        font-size: 13.5px !important;
    }

    /* Full-width CTA Action Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #1B4332 0%, #2D6A4F 60%, #40916C 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 999px !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
        font-size: 14.5px !important;
        font-family: 'Inter', sans-serif !important;
        letter-spacing: 0.02em !important;
        box-shadow: 0 4px 14px rgba(27, 67, 50, 0.22) !important;
        transition: transform 0.18s ease, box-shadow 0.18s ease, opacity 0.18s ease !important;
        width: 100% !important;
        margin-top: 8px !important;
        cursor: pointer !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 22px rgba(27, 67, 50, 0.30) !important;
        color: #FFFFFF !important;
        opacity: 0.98 !important;
    }

    .stButton > button:active {
        transform: translateY(0px) !important;
        box-shadow: 0 2px 8px rgba(27, 67, 50, 0.20) !important;
    }

    /* Logout Button */
    div[class*="st-key-logout_btn"] > button {
        background: linear-gradient(135deg, #9D2B2B 0%, #C1502E 100%) !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 14px rgba(157, 43, 43, 0.20) !important;
        margin-top: 14px !important;
    }

    div[class*="st-key-logout_btn"] > button:hover {
        background: linear-gradient(135deg, #B53333 0%, #D85D3B 100%) !important;
        box-shadow: 0 8px 22px rgba(157, 43, 43, 0.28) !important;
    }

    /* Logged-In Profile View inside Card */
    .user-profile-header {
        text-align: center;
        padding: 6px 0 14px 0;
    }

    .user-avatar-circle {
        width: 52px;
        height: 52px;
        border-radius: 50%;
        background: #D8F3DC;
        border: 1.5px solid #95D5B2;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 24px;
        margin-bottom: 10px;
        color: #1B4332;
    }

    .user-greeting-title {
        font-family: 'Poppins', sans-serif !important;
        font-size: 20px !important;
        font-weight: 700 !important;
        color: #1B4332 !important;
        margin: 0 0 2px 0 !important;
    }

    .user-greeting-sub {
        font-size: 12.5px !important;
        color: #647A6E !important;
        margin: 0 !important;
    }

    .user-profile-meta-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 8px;
        margin: 14px 0;
    }

    .meta-item {
        background: #F7FAF8;
        border: 1px solid rgba(27, 67, 50, 0.08);
        border-radius: 10px;
        padding: 10px 12px;
        display: flex;
        flex-direction: column;
    }

    .meta-label {
        font-size: 10.5px;
        color: #72887C;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        font-weight: 600;
    }

    .meta-value {
        font-family: 'Poppins', sans-serif;
        font-size: 14px;
        font-weight: 700;
        color: #1B4332;
        margin-top: 2px;
        word-break: break-word;
    }

    /* Alerts */
    div[data-testid="stAlert"] {
        background: #FFFFFF !important;
        border: 1px solid rgba(27, 67, 50, 0.10) !important;
        border-radius: 10px !important;
        padding: 10px 14px !important;
        margin-top: 12px !important;
    }
    /* ==================================================
    PREDICTION HISTORY
    ================================================== */

    .prediction-history-title {
        font-family: 'Poppins', sans-serif !important;
        font-size: 22px !important;
        font-weight: 700 !important;
        color: #1B4332 !important;
        margin: 24px 0 14px 0 !important;
        letter-spacing: -0.01em !important;
    }

    /* Remove Streamlit's default blue info styling */
    div[data-testid="stAlert"] {
        background: #F1F8F3 !important;
        border: 1px solid #B7DEC4 !important;
        border-left: 4px solid #2D6A4F !important;
        border-radius: 12px !important;
        color: #1B4332 !important;
    }

    /* Text inside the info box */
    div[data-testid="stAlert"] p {
        color: #2D6A4F !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        line-height: 1.5 !important;
    }

    /* Prediction dataframe */
    div[data-testid="stDataFrame"] {
        border: 1px solid rgba(27, 67, 50, 0.12) !important;
        border-radius: 12px !important;
        overflow: hidden !important;
    }

    /* Make dataframe text readable */
    div[data-testid="stDataFrame"] iframe {
        font-family: 'Inter', sans-serif !important;
    }

    /* Mobile responsive adaptations */
    @media (max-width: 768px) {
        .block-container {
            padding-top: 1rem !important;
            padding-bottom: 1.5rem !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
        .brand-hero-panel {
            padding: 8px 4px 18px 4px !important;
            text-align: center !important;
            align-items: center !important;
        }
        .brand-eyebrow {
            align-self: center !important;
        }
        .brand-main-title {
            font-size: 26px !important;
        }
        .brand-lead-sub {
            font-size: 14px !important;
        }
        .brand-desc {
            font-size: 13px !important;
            margin-bottom: 12px !important;
        }
        .feature-pills-list {
            display: none !important;
        }
        div[data-testid="stVerticalBlockBorderWrapper"][class*="st-key-auth_card"],
        div[data-testid="stVerticalBlock"][class*="st-key-auth_card"],
        div[class*="st-key-auth_card"] {
            padding: 22px 18px !important;
            border-radius: 16px !important;
        }
    }
    </style>
    """

    st.markdown(_static_css, unsafe_allow_html=True)

    #Resposive two colum layout 

    col_brand, col_auth = st.columns([1.15, 1.0], gap="large")

    # ---------------- LEFT: BRANDING & VALUE PROPOSITION ----------------
    with col_brand:
        st.markdown(
            """
            <div class="brand-hero-panel">
                <div class="brand-eyebrow">
                    <span class="brand-dot"></span> Plant Health Intelligence
                </div>
                <h1 class="brand-main-title">🌿 PlantGuard AI</h1>
                <p class="brand-lead-sub">
                    AI-powered plant health analysis
                </p>
                <p class="brand-desc">
                    Protect your plants with intelligent disease detection, real-time weather risk analysis, and deep learning diagnostics.
                </p>
                <div class="feature-pills-list">
                    <div class="feature-pill-item">
                        <div class="feature-icon">🔬</div>
                        <div class="feature-text">
                            <strong>Instant Diagnosis</strong>
                            <span>High-accuracy classification across crop conditions</span>
                        </div>
                    </div>
                    <div class="feature-pill-item">
                        <div class="feature-icon">🌦️</div>
                        <div class="feature-text">
                            <strong>Weather Risk Engine</strong>
                            <span>Pathogen microclimate forecasting & alerts</span>
                        </div>
                    </div>
                    <div class="feature-pill-item">
                        <div class="feature-icon">🔥</div>
                        <div class="feature-text">
                            <strong>Grad-CAM Visuals</strong>
                            <span>Explainable AI leaf lesion attention maps</span>
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ---------------- RIGHT: AUTHENTICATION CARD ----------------
    with col_auth:
        with st.container(key="auth_card"):

            # ---------------- STATE: LOGGED IN ----------------
            if st.session_state.get("logged_in"):
                user_name = st.session_state.get("user_name", "User")
                user_id = st.session_state.get("user_id", "—")

                st.markdown(
                    f"""
                    <div class="user-profile-header">
                        <div class="user-avatar-circle">🌿</div>
                        <h2 class="user-greeting-title">Welcome back, {user_name}!</h2>
                        <p class="user-greeting-sub">Authenticated session active</p>
                    </div>
                    <div class="user-profile-meta-grid">
                        <div class="meta-item">
                            <span class="meta-label">User Name</span>
                            <span class="meta-value">{user_name}</span>
                        </div>
                        <div class="meta-item">
                            <span class="meta-label">User ID</span>
                            <span class="meta-value">#{user_id}</span>
                        </div>
                        <div class="meta-item">
                            <span class="meta-label">Status</span>
                            <span class="meta-value" style="color: #2D6A4F;">Active</span>
                        </div>
                        <div class="meta-item">
                            <span class="meta-label">Access</span>
                            <span class="meta-value">Full</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.success(f"Signed in as **{user_name}**.")

                st.markdown(
                    """
                    <div class="prediction-history-title">
                        🌿 Your Prediction History
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                predictions = get_user_predictions(
                    st.session_state["user_id"]
                )

                if predictions:
                    st.dataframe(predictions, use_container_width=True)
                else:
                    st.info("You don't have any predictions yet.")

                if st.button("Sign Out", key="logout_btn", use_container_width=True):
                    st.session_state["logged_in"] = False
                    st.session_state["user_id"] = None
                    st.session_state["user_name"] = None
                    st.rerun()

            # ---------------- STATE: AUTHENTICATION (SIGN IN / CREATE ACCOUNT) ----------------
            else:
                mode = st.radio(
                    "Auth Mode",
                    ["Sign In", "Create Account"],
                    horizontal=True,
                    label_visibility="collapsed"
                )

                # ---------------- SIGN IN MODE ----------------
                if mode == "Sign In":
                    st.markdown(
                        """
                        <div class="auth-form-header">
                            <h2 class="auth-form-title">Welcome back</h2>
                            <p class="auth-form-sub">Sign in to access your farm diagnostics</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    email = st.text_input("Email", placeholder="name@example.com", key="login_email")
                    password = st.text_input("Password", type="password", placeholder="Enter your password", key="login_password")

                    if st.button("Sign in  →", key="login_submit_btn", use_container_width=True):
                        if not email.strip() or not password:
                            st.warning("Please enter both email and password.")
                        else:
                            user = login_user(email.strip(), password)

                            if user:
                                st.session_state["logged_in"] = True
                                st.session_state["user_id"] = user["id"]
                                st.session_state["user_name"] = user["name"]
                                st.rerun()
                            else:
                                st.error("Invalid email or password. Please try again.")

                # ---------------- CREATE ACCOUNT MODE ----------------
                else:
                    st.markdown(
                        """
                        <div class="auth-form-header">
                            <h2 class="auth-form-title">Create an account</h2>
                            <p class="auth-form-sub">Start protecting your crops with AI</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    name = st.text_input(
                        "Full Name",
                        placeholder="e.g. Alex Johnson",
                        key="signup_name"
                    )

                    email = st.text_input(
                        "Email",
                        placeholder="name@gmail.com",
                        key="signup_email"
                    )

                    password = st.text_input(
                        "Password",
                        type="password",
                        placeholder="Create a secure password",
                        key="signup_password"
                    )

                    if st.button(
                        "Create Account  →",
                        key="signup_submit_btn",
                        use_container_width=True
                    ):

                        name = name.strip()
                        email = email.strip().lower()

                        # Check empty fields
                        if not name or not email or not password:
                            st.warning("Please fill in all fields to create your account.")

                        # Validate Gmail
                        elif not is_valid_email(email):
                            st.error("Please enter a valid Gmail address (example@gmail.com).")

                        # Validate password
                        elif len(password) < 6:
                            st.error("Password must contain at least 6 characters.")

                        # Create account
                        else:
                            success, result = create_user(
                                name,
                                email,
                                password
                            )

                            if success:
                                # Automatically log in the new user
                                st.session_state["logged_in"] = True
                                st.session_state["user_id"] = result["id"]
                                st.session_state["user_name"] = result["name"]

                                st.rerun()

                            else:
                                st.error(result)
if __name__ == "__main__":
    show_auth()