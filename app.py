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
    """
    **NLP-based commodity classification using TF-IDF and Linear SVM**
    
    Enter a commodity description below to predict its **HS Chapter**.
    """
)

st.divider()

# --------------------------------------------------
# User Input
# --------------------------------------------------
description = st.text_area(
    "Enter Commodity Description",
    placeholder="Example: Cotton woven fabric",
    height=120
)

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if st.button("🔍 Predict HS Chapter", use_container_width=True):

    if not description.strip():
        st.warning("Please enter a commodity description.")
    else:
        prediction = model.predict([description])[0]

        st.success(f"Predicted HS Chapter: **{prediction}**")

        st.info(
            "This model predicts the 2-digit HS Chapter based on the "
            "commodity description."
        )

# --------------------------------------------------
# Example Inputs
# --------------------------------------------------
st.divider()

st.subheader("💡 Try an Example")

examples = [
    "Live sheep",
    "Live goats",
    "Cotton woven fabric",
    "Iron or steel products",
    "Fresh fish",
    "Plastic household articles"
]

for example in examples:
    st.write(f"• {example}")

# --------------------------------------------------
# Project Information
# --------------------------------------------------
st.divider()

st.caption(
    "Model: Tuned Linear SVM | Features: TF-IDF | "
    "Test Accuracy: 82.01% | Macro F1: 79.04%"
)
