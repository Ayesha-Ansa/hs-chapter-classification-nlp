# HS4 Code Prediction

### Product Classification using Word + Character TF-IDF and Calibrated Linear SVM

An NLP-based machine learning system that predicts the **4-digit Harmonized System (HS4) heading** for a product description.

The project is designed as a **human-in-the-loop classification assistant**: instead of treating the prediction as an automatic customs decision, the model provides a ranked **Top-3 shortlist with confidence scores** that can support a classification reviewer.

---

## Project Overview

Product classification for international trade can involve reviewing detailed product descriptions and mapping them to the appropriate tariff heading.

This project explores how machine learning can learn patterns from historical tariff-classification examples and predict the most likely **HS4 code** from a new product description.

The core pipeline is:

```text
Product Description
        ↓
Text Cleaning
        ↓
Word TF-IDF + Character TF-IDF
        ↓
Calibrated Linear SVM
        ↓
HS4 Prediction
        ↓
Top-3 Codes + Confidence Scores
```

The model uses both **word-level** and **character-level** text features so it can capture broader product meaning as well as technical terminology and word patterns that can be important in tariff classification.

---

## Objectives

The main objectives of this project are to:

* Build an NLP classification model for HS4 prediction.
* Convert product descriptions into numerical features using TF-IDF.
* Combine word-level and character-level features.
* Train a class-balanced Linear SVM.
* Calibrate the classifier to produce probability estimates.
* Evaluate both **Top-1** and **Top-3** performance.
* Analyze incorrect predictions.
* Provide a ranked shortlist that can support human review.

---

## Dataset

The project uses the publicly available dataset:

**`Dayanand314Krishna/cross_rulings_hts_dataset_for_tariffs`**

Source: [Hugging Face](https://huggingface.co/datasets/Dayanand314Krishna/cross_rulings_hts_dataset_for_tariffs)

The original dataset contains conversational examples stored in a `messages` field.

Each example generally contains:

* A user query describing a product or classification case.
* An assistant response containing one or more HTS codes and supporting reasoning.

The project extracts the **user's product description** as the input and the associated **4-digit HS heading** as the target.

### Dataset preparation

The original dataset contains:

* **18,254** raw records
* **1** source column: `messages`

After parsing and cleaning, the final dataset contains:

* **14,880 usable records**
* **252 HS4 classes**
* **11,904 training samples**
* **2,976 test samples**

---

## Data Cleaning

Several preprocessing steps are applied before training:

1. Extract the user description and HS code from the conversation structure.
2. Convert the target to a 4-digit HS heading.
3. Normalize text to lowercase.
4. Remove records with missing or invalid labels.
5. Remove very short descriptions.
6. Remove conflicting examples where the exact same text is associated with different HS4 labels.
7. Remove exact duplicate text-label pairs.
8. Remove extremely rare HS4 classes with fewer than 10 examples.
9. Perform a stratified train/test split.

The test set is kept separate from model training and is used only for final evaluation.

---

## Model Architecture

### 1. Word-Level TF-IDF

The first feature extractor represents the product description using word-level TF-IDF features.

Configuration:

```python
ngram_range=(1, 2)
min_df=2
max_features=30000
sublinear_tf=True
```

This allows the model to learn:

* Individual terms
* Two-word combinations
* Important product terminology

For example:

```text
cotton
summer dress
stainless steel
rubber parts
```

---

### 2. Character-Level TF-IDF

Character-level TF-IDF is added alongside the word features.

Configuration:

```python
analyzer="char_wb"
ngram_range=(3, 5)
min_df=2
max_features=30000
sublinear_tf=True
```

Character features can capture useful word patterns and technical terminology, including variations in terms such as:

```text
woven
knitted
crocheted
polyester
polyethylene
stainless
plated
```

This is particularly useful when classification depends on specific terminology, word endings, or technical product language.

---

### 3. Feature Combination

The word and character features are combined using `FeatureUnion`.

```text
Word TF-IDF
      +
Character TF-IDF
      ↓
Combined Feature Matrix
```

This gives the classifier access to both semantic word information and finer-grained character patterns.

---

### 4. Calibrated Linear SVM

The final classifier is a **Linear Support Vector Machine** with:

```python
C=1.0
class_weight="balanced"
max_iter=5000
```

Because a standard Linear SVM produces decision scores rather than probabilities, the model is wrapped with:

```python
CalibratedClassifierCV(
    LinearSVC(...),
    cv=3,
    method="sigmoid"
)
```

Calibration converts the model's raw decision scores into probability estimates that can be used to produce a ranked Top-3 prediction list.

---

## Why Calibrated SVM?

Linear SVM is well suited to high-dimensional sparse text representations such as TF-IDF.

The calibration layer adds an additional benefit: instead of returning only a predicted class, the model can provide confidence estimates for multiple possible HS4 headings.

This supports the project's human-review workflow.

---

## Final Model Results

The final model was evaluated on **2,976 unseen test samples**.

| Metric             |     Result |
| ------------------ | ---------: |
| **Top-1 Accuracy** | **69.62%** |
| **Top-3 Accuracy** | **81.69%** |
| **Test Samples**   |  **2,976** |
| **Correct Top-1**  |  **2,072** |
| **Wrong Top-1**    |    **904** |
| **HS4 Classes**    |    **252** |

The model correctly identified the exact HS4 heading as its first prediction for **2,072 out of 2,976** test cases.

For **Top-3 evaluation**, the correct HS4 heading appeared among the model's three highest-ranked predictions in **81.69%** of test cases.

This Top-3 capability is particularly relevant for a human-review workflow because the system can present multiple likely classifications rather than forcing a single automated decision.

---

## Example Prediction

Example input:

```text
Women's 100% woven cotton summer dress with floral embroidery
```

The final model produced:

```text
Top-3 Predictions:

1. HS4: 6204 (24.27%)
2. HS4: 6104 (11.15%)
3. HS4: 6110 (4.56%)
```

In this example, the model correctly placed **6204** at the top of its predictions, demonstrating the value of combining word-level and character-level information for distinguishing closely related product descriptions.

---

## Error Analysis

The project also generates:

```text
wrong_predictions.csv
```

This file contains test cases where the model's Top-1 prediction did not match the actual HS4 heading.

Out of **2,976** test cases:

* **2,072** were correctly classified at Top-1.
* **904** were incorrectly classified at Top-1.

The error analysis helps identify where the classifier struggles, particularly when:

* Multiple HS4 headings have similar product descriptions.
* Important classification attributes are subtle.
* A class has relatively few training examples.
* Similar terminology occurs across different tariff headings.

For example, distinctions such as **woven vs. knitted apparel** can be challenging when one class has substantially more training examples than another.

---

## Project Files

```text
hs4-code-prediction/
│
├── hs4_classifier.ipynb
├── README.md
├── requirements.txt
│
├── final_hs4_calibrated_svm.pkl
├── model_info.pkl
│
├── test_predictions.csv
└── wrong_predictions.csv
```

### Important files

**`hs4_classifier.ipynb`**
Complete data preparation, model training, evaluation, and prediction workflow.

**`final_hs4_calibrated_svm.pkl`**
Serialized trained machine learning pipeline containing the TF-IDF feature engineering and calibrated SVM classifier.

**`model_info.pkl`**
Stores model configuration, class information, and evaluation metadata.

**`test_predictions.csv`**
Predictions for the complete held-out test set.

**`wrong_predictions.csv`**
Subset of test examples where the Top-1 prediction was incorrect.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/hs4-code-prediction.git
cd hs4-code-prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

The complete workflow is available in:

```text
hs4_classifier.ipynb
```

Run the notebook from top to bottom to:

1. Load the dataset.
2. Parse the conversational records.
3. Clean and prepare the data.
4. Create the train/test split.
5. Train the TF-IDF + calibrated SVM model.
6. Evaluate Top-1 and Top-3 accuracy.
7. Generate the classification report.
8. Create prediction and error-analysis CSV files.
9. Save the trained model.
10. Test a new product description.

---

## Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Hugging Face Datasets**
* **Joblib**
* **TF-IDF**
* **Linear SVM**
* **Probability Calibration**
* **Jupyter / Google Colab**

---

## Key Machine Learning Concepts Demonstrated

This project demonstrates practical understanding of:

* Natural Language Processing
* Text preprocessing
* TF-IDF feature engineering
* Word n-grams
* Character n-grams
* Sparse feature representations
* Linear Support Vector Machines
* Class imbalance
* Probability calibration
* Stratified train/test splitting
* Multi-class classification
* Top-K evaluation
* Classification reports
* Error analysis
* Model serialization

---

## Limitations

This project is a machine learning classification experiment and **not a substitute for official customs or tariff-classification guidance**.

Important limitations include:

* The dataset contains a highly uneven distribution across HS4 classes.
* Some classes have relatively few examples.
* Certain descriptions can legitimately be difficult to distinguish.
* Some source records contain more than one product or more than one tariff code.
* The current extraction process uses the first HS4 code identified in the assistant response.
* The model operates at the **4-digit HS heading level**, not the complete country-specific tariff code level.

The system should therefore be viewed as a **decision-support tool for human review**, not an automated customs classification authority.

---

## Future Improvements

Potential future improvements include:

* Incorporating richer product attributes such as material, function, construction, and intended use.
* Increasing representation for underrepresented HS4 classes.
* Exploring transformer-based text representations.
* Building a human-in-the-loop review interface.
* Adding explanation features showing which product terms influenced the prediction.
* Extending the system toward deeper HS hierarchy levels where reliable training data is available.

---

## Project Outcome

The final model achieved:

* **69.62% Top-1 accuracy**
* **81.69% Top-3 accuracy**

across **252 HS4 classes** and **2,976 unseen test samples**.

The project demonstrates how traditional NLP and machine learning techniques can be combined to build a practical product-classification assistant from tariff-classification examples.

Rather than relying on a single predicted label, the system produces a **ranked Top-3 shortlist with calibrated confidence scores**, making it more suitable for workflows where a human reviewer remains responsible for the final classification decision.

---

## Author

**Ayesha Ansari**

Data Science & AI | Machine Learning | NLP
