# ============================================================
# HS4 CODE PREDICTION APP
# TF-IDF + Calibrated Linear SVM
# Includes a conservative low-confidence safeguard
# ============================================================

import os
import urllib.request

import joblib
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="HS4 Code Predictor",
    page_icon="📦",
    layout="centered",
)


# ============================================================
# MODEL SETTINGS
# ============================================================

MODEL_FILE = "final_hs4_calibrated_svm.pkl"
MODEL_URL = os.getenv("MODEL_URL", "")

# Provisional safeguard. The app abstains from presenting a result as
# a stronger prediction when the highest class score is below this value.
# This is a portfolio-level guardrail, not a universal HS classification rule.
ABSTAIN_THRESHOLD = float(os.getenv("HS4_ABSTAIN_THRESHOLD", "0.60"))


# ============================================================
# DOWNLOAD MODEL WHEN NEEDED
# ============================================================

def get_model_path():
    if os.path.exists(MODEL_FILE):
        return MODEL_FILE

    if MODEL_URL:
        st.info("Downloading trained model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)
        return MODEL_FILE

    return None


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    model_path = get_model_path()
    if model_path is None:
        return None
    return joblib.load(model_path)


# ============================================================
# HEADER
# ============================================================

st.title("📦 HS4 Code Predictor")
st.markdown(
    """
### Product Classification Assistant

Enter a product description to explore the **three highest-ranked HS4
headings**. When the model's highest score is below the review threshold,
the app will flag the result as uncertain rather than presenting it as a
reliable prediction.

This is a **human-review assistant**, not an automated customs-classification
authority.
"""
)


# ============================================================
# LOAD MODEL
# ============================================================

try:
    model = load_model()
except Exception as exc:
    st.error("Unable to load the trained model.")
    st.code(str(exc))
    st.warning(
        "Make sure the model file and the required Python package versions "
        "are available."
    )
    st.stop()

if model is None:
    st.warning("The trained model is not available.")
    st.info(
        "For local testing, place 'final_hs4_calibrated_svm.pkl' "
        "in the same folder as app.py."
    )
    st.stop()


# ============================================================
# PRODUCT INPUT
# ============================================================

product_text = st.text_area(
    "Enter product description",
    placeholder=(
        "Example: Women's 100% woven cotton summer dress with floral embroidery"
    ),
    height=150,
)


# ============================================================
# HELPER: DISPLAY RANKED CANDIDATES
# ============================================================

def show_candidates(classes, probabilities, indices, heading):
    st.subheader(heading)
    for rank, index in enumerate(indices, start=1):
        code = str(classes[index])
        confidence = float(probabilities[index])

        st.markdown(f"### {rank}. HS4 **{code}**")
        st.progress(min(max(confidence, 0.0), 1.0))
        st.write(f"Model score: **{confidence * 100:.2f}%**")
        if rank < len(indices):
            st.divider()


# ============================================================
# PREDICTION + ABSTENTION SAFEGUARD
# ============================================================

if st.button("Predict HS4 Code", type="primary", use_container_width=True):
    cleaned_text = product_text.strip().lower()

    if not cleaned_text:
        st.warning("Please enter a product description.")
    else:
        try:
            with st.spinner("Analyzing product description..."):
                probabilities = model.predict_proba([cleaned_text])[0]
                classes = model.classes_
                top_indices = probabilities.argsort()[-3:][::-1]
                top_score = float(probabilities[top_indices[0]])

            if top_score < ABSTAIN_THRESHOLD:
                st.error("⚠️ Unable to provide a sufficiently confident prediction")
                st.write(
                    f"The highest model score is **{top_score * 100:.2f}%**, "
                    f"below the app's provisional review threshold of "
                    f"**{ABSTAIN_THRESHOLD * 100:.0f}%**."
                )
                st.warning(
                    "This product may belong to a category the model does not "
                    "support, or the description may be ambiguous or unfamiliar. "
                    "The model cannot determine from its score alone whether a "
                    "correct HS4 class was excluded during training."
                )
                show_candidates(
                    classes,
                    probabilities,
                    top_indices,
                    "Possible matches — treat these as uncertain, not recommendations",
                )
                st.info(
                    "Try adding relevant details such as material, construction, "
                    "intended use, and product form. If the result remains uncertain, "
                    "verify the code using an authoritative tariff source or a "
                    "qualified reviewer."
                )
            else:
                st.warning(
                    "The model found a higher-scoring candidate, but it can still "
                    "be incorrect. Verify the heading before relying on it."
                )
                show_candidates(
                    classes,
                    probabilities,
                    top_indices,
                    "Top-3 HS4 candidate headings",
                )

        except Exception as exc:
            st.error("An error occurred while generating the prediction.")
            st.code(str(exc))


# ============================================================
# MODEL INFORMATION
# ============================================================

with st.expander("About this model and its safeguard"):
    st.write("Model: Word + Character TF-IDF + Calibrated Linear SVM")
    st.write("Supported HS4 classes: 252")
    st.write("Top-1 accuracy on the held-out test split: 69.62%")
    st.write("Top-3 accuracy on the held-out test split: 81.69%")
    st.write("Held-out test samples: 2,976")
    st.write(f"Provisional low-confidence threshold: {ABSTAIN_THRESHOLD:.0%}")
    st.caption(
        "The threshold is a conservative portfolio safeguard, not a guarantee. "
        "A score above the threshold does not prove the prediction is correct, "
        "and a score below it does not prove the true class was excluded. "
        "This app has not been validated for official customs decisions."
    )


# ============================================================
# DISCLAIMER
# ============================================================

st.divider()
st.caption(
    "This tool provides machine-learning predictions for research and "
    "decision-support purposes. Final tariff classification should be verified "
    "by a qualified reviewer using the applicable customs guidance."
)
