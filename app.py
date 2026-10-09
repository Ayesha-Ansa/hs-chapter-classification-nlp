
# ============================================================
# HS4 CODE PREDICTION APP
# Word + Character TF-IDF + Calibrated Linear SVM
# ============================================================

import os
import math
import tempfile
import urllib.request
from pathlib import Path

import joblib
import streamlit as st


# ============================================================
# 1. CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="HS4 Code Predictor",
    page_icon="📦",
    layout="centered",
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_FILE = BASE_DIR / "final_hs4_calibrated_svm.pkl"

# Initial uncertainty threshold.
# This is a provisional setting, not a validated missing-class detector.
UNCERTAINTY_THRESHOLD = 0.30

MAX_DESCRIPTION_LENGTH = 5000


# ============================================================
# 2. GET MODEL DOWNLOAD URL
# ============================================================

def get_model_url():
    """Read the optional model URL from environment or Streamlit secrets."""

    url = os.environ.get("MODEL_URL", "").strip()

    if url:
        return url

    try:
        url = str(st.secrets.get("MODEL_URL", "")).strip()
    except Exception:
        url = ""

    return url


# ============================================================
# 3. DOWNLOAD MODEL IF NECESSARY
# ============================================================

def get_model_path():
    """
    Use the local model if available.
    Otherwise, download it using MODEL_URL.
    """

    if MODEL_FILE.is_file() and MODEL_FILE.stat().st_size > 0:
        return MODEL_FILE

    model_url = get_model_url()

    if not model_url:
        raise FileNotFoundError(
            "The trained model file was not found.\n\n"
            "Expected file: final_hs4_calibrated_svm.pkl\n\n"
            "Place the model beside app.py or configure MODEL_URL "
            "with a direct downloadable model URL."
        )

    temporary_path = None

    try:
        request = urllib.request.Request(
            model_url,
            headers={"User-Agent": "HS4-Code-Predictor"},
        )

        with urllib.request.urlopen(request, timeout=180) as response:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                suffix=".pkl",
                dir=str(BASE_DIR),
                delete=False,
            ) as temporary_file:

                temporary_path = Path(temporary_file.name)

                while True:
                    chunk = response.read(1024 * 1024)

                    if not chunk:
                        break

                    temporary_file.write(chunk)

        if not temporary_path.exists() or temporary_path.stat().st_size == 0:
            raise ValueError("The downloaded model file is empty.")

        # Check the downloaded file before replacing the destination.
        # Only load model files from a trusted source.
        test_model = joblib.load(temporary_path)

        if not hasattr(test_model, "predict_proba"):
            raise TypeError(
                "The downloaded model does not support predict_proba()."
            )

        os.replace(temporary_path, MODEL_FILE)
        temporary_path = None

        return MODEL_FILE

    except Exception as exc:
        raise RuntimeError(
            "The model could not be downloaded. Check MODEL_URL, "
            "the download permissions, and the model file."
        ) from exc

    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass


# ============================================================
# 4. LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    """Load and cache the existing trained model."""

    model_path = get_model_path()
    loaded_model = joblib.load(model_path)

    if not hasattr(loaded_model, "predict_proba"):
        raise TypeError(
            "The loaded model does not support predict_proba()."
        )

    if not hasattr(loaded_model, "classes_"):
        raise TypeError(
            "The loaded model does not expose classes_."
        )

    if len(loaded_model.classes_) == 0:
        raise ValueError("The model has no trained classes.")

    return loaded_model


# ============================================================
# 5. APP HEADER
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
# 6. LOAD THE MODEL
# ============================================================

try:
    model = load_model()

except Exception as exc:
    st.error(
        "The trained model could not be loaded. "
        "The application cannot make predictions until this is fixed."
    )

    with st.expander("Technical error details", expanded=True):
        st.code(f"{type(exc).__name__}: {exc}")

    st.stop()


# ============================================================
# 7. PRODUCT INPUT
# ============================================================

product_text = st.text_area(
    "Enter product description",
    placeholder=(
        "Example: Women's 100% woven cotton summer dress "
        "with floral embroidery"
    ),
    height=150,
    max_chars=MAX_DESCRIPTION_LENGTH,
)


# ============================================================
# 8. PREDICTION
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

                probabilities = model.predict_proba([cleaned_text])[0]
                classes = model.classes_

                if len(probabilities) != len(classes):
                    raise ValueError(
                        "The prediction scores do not match "
                        "the trained model classes."
                    )

                if len(probabilities) == 0:
                    raise ValueError(
                        "The model returned no prediction scores."
                    )

                if not all(
                    math.isfinite(float(score))
                    for score in probabilities
                ):
                    raise ValueError(
                        "The model returned invalid prediction scores."
                    )

                # Rank from highest to lowest score.
                top_indices = probabilities.argsort()[::-1][:3]

                top_score = float(probabilities[top_indices[0]])
                predicted_code = str(classes[top_indices[0]])

            # ------------------------------------------------
            # 9. UNCERTAINTY WARNING
            # ------------------------------------------------

            if top_score < UNCERTAINTY_THRESHOLD:

                st.warning(
                    "⚠️ Uncertain classification — manual review recommended."
                )

                st.write(
                    f"The highest model score is "
                    f"**{top_score * 100:.2f}%**, below the provisional "
                    f"{UNCERTAINTY_THRESHOLD * 100:.0f}% review threshold."
                )

                st.info(
                    "The model may not recognize this product description "
                    "well enough to make a reliable prediction. This may "
                    "be caused by an unfamiliar description, ambiguity, "
                    "or a product class not represented in training. "
                    "The score alone cannot establish which explanation "
                    "is correct."
                )

                st.subheader("Low-confidence candidates")

                st.caption(
                    "These are model guesses, not reliable recommendations. "
                    "Verify the correct heading independently."
                )

            else:

                st.success("Prediction generated.")

                st.subheader("Top-3 HS4 Predictions")

            # ------------------------------------------------
            # 10. DISPLAY TOP THREE CANDIDATES
            # ------------------------------------------------

            for rank, index in enumerate(top_indices, start=1):

                code = str(classes[index])
                score = float(probabilities[index])

                st.markdown(f"### {rank}. HS4 **{code}**")

                st.write(f"Model score: **{score * 100:.2f}%**")

                st.progress(max(0.0, min(1.0, score)))

                if rank < len(top_indices):
                    st.divider()

            # ------------------------------------------------
            # 11. IMPORTANT LIMITATION
            # ------------------------------------------------

            st.caption(
                "Model scores rank the classes learned during training. "
                "They do not guarantee correctness. A high score does not "
                "prove that the product's true class is supported, and a "
                "low score does not prove that its true class was excluded."
            )

        except Exception as exc:

            st.error("An error occurred while generating the prediction.")

            with st.expander("Technical error details", expanded=True):
                st.code(f"{type(exc).__name__}: {exc}")


# ============================================================
# 12. MODEL INFORMATION
# ============================================================

with st.expander("About this model"):

    st.write(
        "Model: Word + Character TF-IDF + Calibrated Linear SVM"
    )

    st.write(f"Classes learned during training: {len(model.classes_)}")

    st.write(
        f"Provisional uncertainty threshold: "
        f"{UNCERTAINTY_THRESHOLD * 100:.0f}%"
    )

    st.caption(
        "The threshold has not yet been validated as an unsupported-class "
        "detector. Prediction scores may not be reliable probabilities "
        "for unfamiliar products."
    )


# ============================================================
# 13. DISCLAIMER
# ============================================================

st.divider()

st.caption(
    "For research and decision support only. Verify the final HS4 "
    "classification against authoritative tariff guidance and "
    "qualified review."
)
