
# ============================================================
# HS4 CODE PREDICTION APP
# TF-IDF + Calibrated Linear SVM
# ============================================================

import os
import urllib.request
from pathlib import Path

import joblib
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="HS4 Code Predictor",
    page_icon="📦",
    layout="centered",
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_FILE = BASE_DIR / "final_hs4_calibrated_svm.pkl"

# Keep your existing Streamlit secret/environment configuration.
MODEL_URL = os.getenv("MODEL_URL", "").strip()


# ============================================================
# MODEL DOWNLOAD
# ============================================================

def get_model_path():
    """
    Return the local model path.
    If the model is not present locally, download it using MODEL_URL.
    """

    if MODEL_FILE.is_file():
        return MODEL_FILE

    if not MODEL_URL:
        raise FileNotFoundError(
            "The trained model file was not found. "
            "Place final_hs4_calibrated_svm.pkl beside app.py "
            "or configure MODEL_URL."
        )

    try:
        st.info("Downloading the trained model...")

        request = urllib.request.Request(
            MODEL_URL,
            headers={"User-Agent": "HS4-Code-Predictor"},
        )

        with urllib.request.urlopen(request, timeout=120) as response:
            model_bytes = response.read()

        if not model_bytes:
            raise ValueError("The downloaded model file is empty.")

        # Write only after the download completes.
        temporary_file = MODEL_FILE.with_suffix(".tmp")

        try:
            temporary_file.write_bytes(model_bytes)
            temporary_file.replace(MODEL_FILE)
        finally:
            if temporary_file.exists():
                temporary_file.unlink()

        return MODEL_FILE

    except Exception as exc:
        raise RuntimeError(
            "Unable to download the trained model. "
            "Check MODEL_URL and the model download link."
        ) from exc


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    """Load and cache the existing trained model."""

    model_path = get_model_path()
    loaded_model = joblib.load(model_path)

    # Validate the minimum interface required by the app.
    if not hasattr(loaded_model, "predict_proba"):
        raise TypeError(
            "The loaded model does not support predict_proba()."
        )

    if not hasattr(loaded_model, "classes_"):
        raise TypeError(
            "The loaded model does not expose classes_."
        )

    if len(loaded_model.classes_) == 0:
        raise ValueError("The model contains no trained classes.")

    return loaded_model


# ============================================================
# APP HEADER
# ============================================================

st.title("📦 HS4 Code Predictor")

st.markdown(
    """
    ### Product Classification Assistant

    Enter a product description to receive the **Top-3 HS4
    candidate headings** with model scores.

    This application is a machine-learning decision-support tool.
    Always verify tariff classifications before relying on them.
    """
)


# ============================================================
# LOAD THE EXISTING TRAINED MODEL
# ============================================================

try:
    model = load_model()

except Exception:
    st.error(
        "Unable to load the trained model. "
        "Check the model file, download configuration, "
        "and installed package versions."
    )

    with st.expander("Technical details"):
        st.exception(__import__("sys").exc_info()[1])

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
    height=150,
    max_chars=5000,
)


# ============================================================
# PREDICTION
# ============================================================

if st.button(
    "Predict HS4 Code",
    type="primary",
    use_container_width=True,
):

    cleaned_text = product_text.strip().lower()

    if not cleaned_text:
        st.warning("Please enter a product description.")

    else:
        try:
            with st.spinner("Analyzing product description..."):

                probabilities = model.predict_proba(
                    [cleaned_text]
                )[0]

                classes = model.classes_

                # Rank model outputs from highest to lowest.
                top_indices = probabilities.argsort()[::-1][:3]

                if len(probabilities) != len(classes):
                    raise ValueError(
                        "The number of prediction scores does not "
                        "match the number of trained classes."
                    )

                if not all(
                    __import__("math").isfinite(float(probabilities[i]))
                    for i in top_indices
                ):
                    raise ValueError(
                        "The model returned invalid prediction scores."
                    )

            # ------------------------------------------------
            # RESULTS
            # ------------------------------------------------

            st.success("Prediction generated.")

            st.subheader("Top-3 HS4 Predictions")

            for rank, index in enumerate(top_indices, start=1):

                code = str(classes[index])
                score = float(probabilities[index])

                st.markdown(
                    f"### {rank}. HS4 **{code}**"
                )

                st.progress(
                    max(0.0, min(1.0, score))
                )

                st.write(
                    f"Model score: **{score * 100:.2f}%**"
                )

                if rank < len(top_indices):
                    st.divider()

            # ------------------------------------------------
            # IMPORTANT MODEL LIMITATION
            # ------------------------------------------------

            st.caption(
                "These scores rank the classes learned during "
                "training. They do not guarantee correctness or "
                "prove that the correct HS4 class is supported. "
                "The app cannot identify a missing true class "
                "from the model score alone."
            )

        except Exception:
            st.error(
                "Prediction failed. Please try again or check "
                "the technical details below."
            )

            with st.expander("Technical details"):
                st.exception(__import__("sys").exc_info()[1])


# ============================================================
# MODEL INFORMATION
# ============================================================

with st.expander("About this model"):

    st.write(
        "Model: Word + Character TF-IDF + Calibrated Linear SVM"
    )

    st.write(f"Supported trained classes: {len(model.classes_)}")

    st.write("Reported Top-1 accuracy: 69.62%")

    st.write("Reported Top-3 accuracy: 81.69%")

    st.write("Evaluation test samples: 2,976")

    st.caption(
        "Reported metrics describe the existing evaluation "
        "results; they are not a guarantee of live prediction quality."
    )


# ============================================================
# DISCLAIMER
# ============================================================

st.divider()

st.caption(
    "This tool provides machine-learning predictions for research "
    "and decision-support purposes. The predicted heading may be "
    "incorrect. Verify the final classification using applicable "
    "official tariff guidance and qualified review."
)
