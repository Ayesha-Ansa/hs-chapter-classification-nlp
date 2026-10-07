# ============================================================
# HS Chapter Classification — Streamlit App
# Loads the trained TF-IDF + Linear SVM pipeline and predicts
# the 2-digit HS Chapter for a user-entered commodity description.
# ============================================================

import streamlit as st
import joblib
import numpy as np

# ------------------------------------------------------------
# Page config
# ------------------------------------------------------------
st.set_page_config(
    page_title="HS Chapter Classifier",
    page_icon="📦",
    layout="centered"
)

# ------------------------------------------------------------
# Load the trained model (cached so it only loads once)
# ------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("hs_chapter_nlp_classifier.pkl")

try:
    model = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

# ------------------------------------------------------------
# Optional: HS Chapter number -> short human-readable name
# Used only to make results easier to read; prediction itself
# is based purely on the model.
# ------------------------------------------------------------
CHAPTER_NAMES = {
    "01": "Live animals", "02": "Meat and edible meat offal",
    "03": "Fish, crustaceans, molluscs", "04": "Dairy, eggs, honey",
    "05": "Products of animal origin, nes", "06": "Live trees, plants, cut flowers",
    "07": "Edible vegetables and roots", "08": "Edible fruit and nuts",
    "09": "Coffee, tea, mate, spices", "10": "Cereals",
    "11": "Milling products, malt, starches", "12": "Oil seeds, oleaginous fruits",
    "13": "Lac, gums, resins, vegetable saps", "14": "Vegetable plaiting materials",
    "15": "Animal/vegetable fats and oils", "16": "Meat/fish food preparations",
    "17": "Sugars and sugar confectionery", "18": "Cocoa and cocoa preparations",
    "19": "Cereal, flour, starch preparations", "20": "Vegetable/fruit/nut preparations",
    "21": "Miscellaneous edible preparations", "22": "Beverages, spirits, vinegar",
    "23": "Food industry residues, animal fodder", "24": "Tobacco products",
    "25": "Salt, sulphur, earth, stone, cement", "26": "Ores, slag, ash",
    "27": "Mineral fuels, oils", "28": "Inorganic chemicals",
    "29": "Organic chemicals", "30": "Pharmaceutical products",
    "31": "Fertilizers", "32": "Tanning/dyeing extracts, pigments",
    "33": "Essential oils, perfumes, cosmetics", "34": "Soaps, waxes, candles",
    "35": "Albuminoids, modified starches, glues", "36": "Explosives, pyrotechnics, matches",
    "37": "Photographic/cinematographic goods", "38": "Miscellaneous chemical products",
    "39": "Plastics and articles thereof", "40": "Rubber and articles thereof",
    "41": "Raw hides and skins", "42": "Leather articles",
    "43": "Furskins and artificial fur", "44": "Wood and articles of wood",
    "45": "Cork and articles of cork", "46": "Manufactures of plaiting material",
    "47": "Wood pulp, fibrous cellulosic material", "48": "Paper, paperboard articles",
    "49": "Printed books, newspapers", "50": "Silk",
    "51": "Wool, animal hair, yarn, fabric", "52": "Cotton",
    "53": "Vegetable textile fibres", "54": "Man-made filaments",
    "55": "Man-made staple fibres", "56": "Wadding, felt, nonwovens",
    "57": "Carpets and textile floor coverings", "58": "Special woven/tufted fabrics, lace",
    "59": "Impregnated/coated textile fabric", "60": "Knitted or crocheted fabric",
    "61": "Apparel, knit or crochet", "62": "Apparel, not knit or crochet",
    "63": "Other made textile articles", "64": "Footwear, gaiters",
    "65": "Headgear", "66": "Umbrellas, walking sticks",
    "67": "Feathers, artificial flowers", "68": "Stone, plaster, cement articles",
    "69": "Ceramic products", "70": "Glass and glassware",
    "71": "Pearls, precious stones, metals", "72": "Iron and steel",
    "73": "Articles of iron or steel", "74": "Copper and articles thereof",
    "75": "Nickel and articles thereof", "76": "Aluminium and articles thereof",
    "78": "Lead and articles thereof", "79": "Zinc and articles thereof",
    "80": "Tin and articles thereof", "81": "Other base metals",
    "82": "Tools, implements, cutlery", "83": "Miscellaneous base metal articles",
    "84": "Nuclear reactors, boilers, machinery", "85": "Electrical/electronic equipment",
    "86": "Railway, tramway equipment", "87": "Vehicles other than railway",
    "88": "Aircraft, spacecraft", "89": "Ships, boats",
    "90": "Optical, photo, medical instruments", "91": "Clocks and watches",
    "92": "Musical instruments", "93": "Arms and ammunition",
    "94": "Furniture, lighting, prefab buildings", "95": "Toys, games, sports requisites",
    "96": "Miscellaneous manufactured articles", "97": "Works of art, antiques"
}

# ------------------------------------------------------------
# UI
# ------------------------------------------------------------
st.title("📦 HS Chapter Classifier")
st.markdown(
    "Predicts the **2-digit HS Chapter** for a commodity description, "
    "using a TF-IDF + Linear SVM model trained on international trade data."
)
st.caption(
    "Note: this predicts the broader HS Chapter, not the full Canadian HTS code."
)

if not model_loaded:
    st.error(
        "Model file `hs_chapter_nlp_classifier.pkl` not found. "
        "Make sure it's in the same folder as this app."
    )
    st.stop()

description = st.text_input(
    "Enter a commodity / product description:",
    placeholder="e.g. cotton woven fabric"
)

if st.button("Predict HS Chapter", type="primary"):
    if not description.strip():
        st.warning("Please enter a description first.")
    else:
        # Primary prediction
        prediction = model.predict([description])[0]
        chapter_name = CHAPTER_NAMES.get(prediction, "Unknown")

        st.success(f"**Predicted HS Chapter: {prediction}** — {chapter_name}")

        # Top-3 candidates using the SVM's decision function
        # (LinearSVC doesn't output probabilities directly, so we rank
        # classes by their decision score as a proxy for confidence)
        try:
            scores = model.decision_function([description])[0]
            classes = model.classes_
            top3_idx = np.argsort(scores)[-3:][::-1]

            st.markdown("**Top 3 candidate chapters (for analyst review):**")
            for idx in top3_idx:
                code = classes[idx]
                name = CHAPTER_NAMES.get(code, "Unknown")
                st.write(f"- Chapter {code} — {name}  (score: {scores[idx]:.2f})")
        except Exception:
            pass  # top-3 is a nice-to-have; skip silently if unavailable

st.divider()
st.caption(
    "This is a prototype decision-support tool and does not replace expert "
    "tariff classification. Built by Ayesha Ansari."
)
