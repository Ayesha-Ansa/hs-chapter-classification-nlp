import streamlit as st
import joblib

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="HS Chapter Classifier",
    page_icon="📦",
    layout="centered"
)

# --------------------------------------------------
# Minimal custom styling
# --------------------------------------------------
st.markdown(
    """
    <style>
    .main > div {
        padding-top: 2rem;
    }
    .stButton button {
        border-radius: 8px;
        font-weight: 600;
    }
    .example-chip {
        display: inline-block;
        background-color: #f0f2f6;
        border-radius: 16px;
        padding: 4px 14px;
        margin: 4px;
        font-size: 0.85rem;
        color: #333;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# Load Trained Model
# --------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("hs_chapter_nlp_classifier.pkl")


model = load_model()

# --------------------------------------------------
# Header
# --------------------------------------------------
st.title("📦 HS Chapter Classification")
st.markdown(
    "**NLP-based commodity classification using TF-IDF and Linear SVM**"
)
st.caption(
    "Predicts the 2-digit HS Chapter from a commodity description. "
    "Not the full Canadian HTS code."
)

# --------------------------------------------------
# Model performance strip
# --------------------------------------------------
col1, col2, col3 = st.columns(3)
col1.metric("Test Accuracy", "82.01%")
col2.metric("Macro F1", "79.04%")
col3.metric("Model", "Linear SVM")

st.divider()

# --------------------------------------------------
# Example chips (click to fill the input)
# --------------------------------------------------
st.markdown("**💡 Try an example:**")

examples = [
    "Live sheep",
    "Live goats",
    "Cotton woven fabric",
    "Iron or steel products",
    "Fresh fish",
    "Plastic household articles"
]

if "description_input" not in st.session_state:
    st.session_state.description_input = ""

example_cols = st.columns(3)
for i, example in enumerate(examples):
    with example_cols[i % 3]:
        if st.button(example, key=f"ex_{i}", use_container_width=True):
            st.session_state.description_input = example

st.write("")

# --------------------------------------------------
# User Input
# --------------------------------------------------
description = st.text_area(
    "Enter Commodity Description",
    value=st.session_state.description_input,
    placeholder="Example: Cotton woven fabric",
    height=100,
    key="description_box"
)

# --------------------------------------------------
# Prediction
# --------------------------------------------------
predict_clicked = st.button("🔍 Predict HS Chapter", use_container_width=True, type="primary")

if predict_clicked:
    if not description.strip():
        st.warning("Please enter a commodity description.")
    else:
        prediction = model.predict([description])[0]

        st.success(f"### Predicted HS Chapter: **{prediction}**")
        st.info(
            "This model predicts the 2-digit HS Chapter based on the "
            "commodity description."
        )

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.divider()
st.caption(
    "Prototype decision-support tool — not a substitute for expert tariff "
    "classification. Built by Ayesha Ansari."
)
