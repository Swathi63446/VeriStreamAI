"""Streamlit UI for Fake News Detection."""

import joblib
import streamlit as st

EXAMPLES = {
    "Economic policy announcement": (
        "Government announces a new economic recovery policy focused on small "
        "business tax relief and increased public infrastructure spending."
    ),
    "Election conspiracy rumor": (
        "A viral post claims election officials secretly replaced all ballot "
        "machines overnight, but the claim cites no official records or evidence."
    ),
    "Health research summary": (
        "Researchers at a public university published a peer-reviewed study "
        "showing early evidence that regular exercise improves sleep quality."
    ),
}


st.set_page_config(
    page_title="VeriStream AI",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&display=swap');

      :root {
        --bg: #f4f7f5;
        --ink: #16251f;
        --muted: #5d6f67;
        --brand: #0f766e;
        --brand-soft: #c7efe9;
        --ok: #1f8a53;
        --warn: #b45309;
        --bad: #b42318;
        --card: #ffffff;
      }

      html, body, [class*="stApp"] {
        font-family: 'Space Grotesk', sans-serif;
        background:
          radial-gradient(circle at 12% 8%, #d8f8ea 0%, transparent 28%),
          radial-gradient(circle at 90% 20%, #dff4ff 0%, transparent 30%),
          var(--bg);
      }

      .block-container {
        padding-top: 1.4rem;
        padding-bottom: 1rem;
      }

      .hero {
        border: 1px solid #d4e8df;
        border-radius: 18px;
        padding: 1.1rem 1.2rem;
        background: linear-gradient(130deg, #ffffff 0%, #ecfaf5 100%);
        box-shadow: 0 8px 30px rgba(11, 86, 70, 0.08);
        margin-bottom: 0.9rem;
      }

      .hero h1 {
        margin: 0;
        line-height: 1.1;
        color: var(--ink);
        font-size: 1.9rem;
        letter-spacing: -0.03em;
      }

      .hero p {
        margin: 0.45rem 0 0 0;
        color: var(--muted);
        font-size: 1rem;
      }

      .tag-row {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
        margin-top: 0.75rem;
      }

      .tag {
        font-size: 0.76rem;
        color: #0a5a50;
        background: var(--brand-soft);
        border: 1px solid #9fded6;
        border-radius: 999px;
        padding: 0.24rem 0.6rem;
        font-weight: 700;
      }

      .result-card {
        border-radius: 14px;
        padding: 1rem;
        border: 1px solid;
      }

      .result-real {
        background: #eafaf1;
        border-color: #a8dec2;
      }

      .result-fake {
        background: #fff1f2;
        border-color: #f3b4ba;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model_and_vectorizer():
    """Load trained model artifacts."""
    try:
        model = joblib.load("./models/classifier.pkl")
        vectorizer = joblib.load("./models/vectorizer.pkl")
        return model, vectorizer
    except FileNotFoundError:
        return None, None


def predict_news(text, model, vectorizer):
    """Predict label and certainty proxy for input text."""
    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)[0]
    margin = model.decision_function(text_vectorized)[0]
    certainty = abs(margin) / (1 + abs(margin))
    return prediction, certainty, margin


def apply_selected_example():
    """Load selected sample text into the input area."""
    st.session_state.user_input = EXAMPLES[st.session_state.example_choice]


def clear_input_text():
    """Clear the input text area."""
    st.session_state.user_input = ""


def certainty_label(certainty):
    """Human-readable certainty bands."""
    if certainty < 0.20:
        return "Low", "The model is near the decision boundary."
    if certainty < 0.50:
        return "Medium", "Prediction is usable but still uncertain."
    return "High", "Prediction is far from the decision boundary."


def main():
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""
    if "example_choice" not in st.session_state:
        st.session_state.example_choice = list(EXAMPLES.keys())[0]

    model, vectorizer = load_model_and_vectorizer()
    if model is None or vectorizer is None:
        st.error(
            "Model files are missing. Run python train.py to create models/classifier.pkl "
            "and models/vectorizer.pkl."
        )
        st.stop()

    with st.sidebar:
        st.markdown("### Model Snapshot")
        st.caption("Pipeline: TF-IDF + PassiveAggressiveClassifier")
        st.metric("Offline Validation Accuracy", "93.05%")
        st.caption(
            "Certainty is margin-based, not a calibrated probability. "
            "Longer, richer text generally improves stability."
        )

    st.markdown(
        """
        <div class="hero">
          <h1>SignalLens News Verifier</h1>
          <p>Paste a claim or article snippet, then score it as likely REAL or FAKE with margin-based certainty insights.</p>
          <div class="tag-row">
            <span class="tag">Fast inference</span>
            <span class="tag">Human-readable certainty</span>
            <span class="tag">Interactive examples</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left_col, right_col = st.columns([2.4, 1.1], gap="large")

    with right_col:
        st.subheader("Actions")
        st.selectbox(
            "Example set",
            options=list(EXAMPLES.keys()),
            key="example_choice",
            label_visibility="collapsed",
        )
        st.button("Load Example", use_container_width=True, on_click=apply_selected_example)
        st.button("Clear Text", use_container_width=True, on_click=clear_input_text)
        analyze_button = st.button("Analyze News", use_container_width=True, type="primary")

    with left_col:
        st.subheader("Article Input")
        st.text_area(
            "Paste article or claim",
            key="user_input",
            height=260,
            placeholder="Paste a complete paragraph for better certainty...",
            label_visibility="collapsed",
        )

    if analyze_button:
        article_text = st.session_state.user_input.strip()
        if not article_text:
            st.warning("Please paste some text before running analysis.")
        else:
            with st.spinner("Scanning language patterns and feature weights..."):
                prediction, certainty, margin = predict_news(article_text, model, vectorizer)

            tone_class = "result-real" if prediction == "REAL" else "result-fake"
            emoji = "✅" if prediction == "REAL" else "⚠️"
            message = (
                "Likely factual reporting style"
                if prediction == "REAL"
                else "Likely misleading or fabricated style"
            )

            st.markdown(
                f"""
                <div class="result-card {tone_class}">
                  <h3 style="margin:0;">{emoji} Prediction: {prediction}</h3>
                  <p style="margin:0.45rem 0 0 0;">{message}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            level, explanation = certainty_label(certainty)
            st.markdown("#### Certainty")
            st.progress(float(certainty))
            st.caption(f"{certainty * 100:.2f}% ({level}) - {explanation}")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Label", prediction)
            m2.metric("Certainty", f"{certainty * 100:.2f}%")
            m3.metric("Characters", str(len(article_text)))
            m4.metric("Words", str(len(article_text.split())))

            with st.expander("Detailed diagnostics"):
                st.write(f"Prediction class: {prediction}")
                st.write(f"Certainty score: {certainty:.4f}")
                st.write(f"Raw decision margin: {margin:.4f}")
                st.write(
                    "Note: this is not a calibrated probability. "
                    "It is a bounded confidence proxy from model margin."
                )

    st.caption("SignalLens prototype | Fake News Detection with scikit-learn")


if __name__ == "__main__":
    main()
