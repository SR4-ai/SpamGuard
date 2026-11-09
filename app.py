import re
import joblib
import streamlit as st

from pathlib import Path


# ==========================================
# 1. APPLICATION CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

st.set_page_config(
    page_title="SpamGuard | SMS Detector",
    page_icon="🛡️",
    layout="centered"
)


# ==========================================
# 2. LOAD TRAINED MODEL
# ==========================================

@st.cache_resource
def load_model():

    model = joblib.load(
        MODEL_DIR / "spam_classifier.pkl"
    )

    vectorizer = joblib.load(
        MODEL_DIR / "tfidf_vectorizer.pkl"
    )

    threshold = joblib.load(
        MODEL_DIR / "spam_threshold.pkl"
    )

    return model, vectorizer, threshold


model, vectorizer, SPAM_THRESHOLD = load_model()


# ==========================================
# 3. TEXT PREPROCESSING
# ==========================================

def preprocess_text(text):

    # Convert text to lowercase
    text = text.lower()

    # Remove punctuation and special characters
    text = re.sub(r"[^a-z0-9\s]", "", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ==========================================
# 4. CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.stApp {
    background-color: #0b1120;
    color: #e5e7eb;
}

.main .block-container {
    max-width: 850px;
    padding-top: 3rem;
    padding-bottom: 2rem;
}

/* Main heading */

.title {
    text-align: center;
    font-size: 46px;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -1px;
}

.subtitle {
    text-align: center;
    color: #94a3b8;
    font-size: 16px;
    margin-top: 5px;
    margin-bottom: 40px;
}

/* Section heading */

.section-heading {
    font-size: 23px;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 10px;
}

/* Result cards */

.result-box {
    padding: 28px;
    border-radius: 15px;
    text-align: center;
    margin-top: 25px;
}

.spam {
    background-color: #451a1a;
    border: 1px solid #ef4444;
}

.ham {
    background-color: #123524;
    border: 1px solid #22c55e;
}

.result-title {
    font-size: 28px;
    font-weight: bold;
    margin-bottom: 8px;
}

.result-description {
    color: #cbd5e1;
    font-size: 15px;
}

/* Footer */

.footer {
    text-align: center;
    color: #64748b;
    font-size: 13px;
    margin-top: 55px;
    padding: 20px;
    border-top: 1px solid #1e293b;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# 5. APPLICATION HEADER
# ==========================================

st.markdown(
    '<div class="title">SpamGuard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-Powered SMS Spam Detection System'
    '</div>',
    unsafe_allow_html=True
)


# ==========================================
# 6. MESSAGE INPUT
# ==========================================

st.markdown(
    '<div class="section-heading">Analyze Your Message</div>',
    unsafe_allow_html=True
)

message = st.text_area(
    "Enter your SMS message below:",
    height=160,
    placeholder="Example: Congratulations! You have won a free prize. Claim now...",
    key="sms_input"
)


# ==========================================
# 7. MESSAGE ANALYSIS
# ==========================================

if st.button(
    "Analyze Message",
    use_container_width=True,
    type="primary"
):

    if not message.strip():

        st.warning("Please enter a message first.")

    else:

        cleaned_message = preprocess_text(message)

        if not cleaned_message:

            st.warning("Please enter meaningful text.")

        else:

            # Convert message into TF-IDF features
            message_vector = vectorizer.transform(
                [cleaned_message]
            )

            # Get spam probability
            spam_index = list(model.classes_).index("spam")

            spam_probability = model.predict_proba(
                message_vector
            )[0][spam_index]

            # Apply selected classification threshold
            if spam_probability >= SPAM_THRESHOLD:

                result = "SPAM"

                result_html = """
                <div class="result-box spam">
                    <div class="result-title">
                        SPAM DETECTED
                    </div>

                    <div class="result-description">
                        This message appears suspicious.
                    </div>
                </div>
                """

            else:

                result = "HAM"

                result_html = """
                <div class="result-box ham">
                    <div class="result-title">
                        SAFE MESSAGE
                    </div>

                    <div class="result-description">
                        This message appears to be a normal SMS.
                    </div>
                </div>
                """

            # Display classification result
            st.html(result_html)

            # Display analysis information
            st.write("")

            st.subheader("Analysis Results")

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Spam Probability",
                    f"{spam_probability * 100:.2f}%"
                )

            with col2:

                st.metric(
                    "Classification",
                    result
                )

            # Probability progress bar
            st.write("Spam Probability")

            st.progress(
                float(spam_probability)
            )

            st.caption(
                f"Classification threshold: "
                f"{SPAM_THRESHOLD * 100:.0f}%"
            )

            st.info(
                "Note: Predictions are based on statistical "
                "patterns learned from the training dataset. "
                "They are not a guarantee of message safety."
            )


# ==========================================
# 8. SAMPLE MESSAGES
# ==========================================

with st.expander("Try sample messages"):

    st.markdown("**Spam example**")

    st.code(
        "Congratulations! You have won a free iPhone. "
        "Claim your prize now!"
    )

    st.markdown("**Normal example**")

    st.code(
        "Hey, are you coming to college tomorrow?"
    )


# ==========================================
# 9. FOOTER
# ==========================================

st.html("""
<div class="footer">
    SpamGuard | Machine Learning Project
    <br><br>
    Built using Python, Scikit-learn and Streamlit
</div>
""")