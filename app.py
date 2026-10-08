# ============================================================
# HS4 CODE PREDICTION APP
# TF-IDF + Calibrated Linear SVM
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
    layout="centered"
)


# ============================================================
# MODEL SETTINGS
# ============================================================

MODEL_FILE = "final_hs4_calibrated_svm.pkl"

# For deployment, set MODEL_URL to the direct download URL
# of your model file.
MODEL_URL = os.getenv("MODEL_URL", "")


# ============================================================
# DOWNLOAD MODEL WHEN NEEDED
# ============================================================

def get_model_path():

    if os.path.exists(MODEL_FILE):
        return MODEL_FILE

    if MODEL_URL:

        st.info("Downloading trained model...")

        urllib.request.urlretrieve(
            MODEL_URL,
            MODEL_FILE
        )

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

Enter a product description to receive the **three most likely
HS4 tariff headings** with calibrated confidence scores.

The system is designed as a **human-review assistant**, not as
an automated customs classification authority.
"""
)


# ============================================================
# LOAD MODEL
# ============================================================

try:

    model = load_model()

except Exception as e:

    st.error("Unable to load the trained model.")

    st.code(str(e))

    st.warning(
        "Make sure the model file and the required Python "
        "package versions are available."
    )

    st.stop()


if model is None:

    st.warning(
        "The trained model is not available."
    )

    st.info(
        "For local testing, place "
        "'final_hs4_calibrated_svm.pkl' "
        "in the same folder as app.py."
    )

    st.stop()


# ============================================================
# PRODUCT INPUT
# ============================================================

product_text = st.text_area(
    "Enter product description",
    placeholder=(
        "Example: Women's 100% woven cotton summer "
        "dress with floral embroidery"
    ),
    height=150
)


# ============================================================
# PREDICTION
# ============================================================

if st.button(
    "Predict HS4 Code",
    type="primary",
    use_container_width=True
):

    if not product_text.strip():

        st.warning(
            "Please enter a product description."
        )

    else:

        with st.spinner("Analyzing product description..."):

            probabilities = model.predict_proba(
                [product_text.strip().lower()]
            )[0]

            classes = model.classes_

            top3_index = (
                probabilities
                .argsort()[-3:][::-1]
            )


        # ----------------------------------------------------
        # RESULTS
        # ----------------------------------------------------

        st.success("Prediction complete!")

        st.subheader("Top-3 HS4 Predictions")

        for rank, index in enumerate(
            top3_index,
            start=1
        ):

            code = classes[index]

            confidence = (
                probabilities[index] * 100
            )

            st.markdown(
                f"### {rank}. HS4 **{code}**"
            )

            st.progress(
                float(probabilities[index])
            )

            st.write(
                f"Confidence: **{confidence:.2f}%**"
            )

            if rank < 3:
                st.divider()


# ============================================================
# MODEL INFORMATION
# ============================================================

with st.expander("About this model"):

    st.write(
        "Model: Word + Character TF-IDF + Calibrated Linear SVM"
    )

    st.write(
        "HS4 classes: 252"
    )

    st.write(
        "Top-1 accuracy: 69.62%"
    )

    st.write(
        "Top-3 accuracy: 81.69%"
    )

    st.write(
        "Test samples: 2,976"
    )


# ============================================================
# DISCLAIMER
# ============================================================

st.divider()

st.caption(
    "This tool provides machine-learning predictions for "
    "research and decision-support purposes. Final tariff "
    "classification should be verified by a qualified reviewer "
    "using the applicable customs guidance."
)
