# HS Chapter Classification Using NLP

### Commodity Description → HS Chapter Prediction

An NLP-based multiclass classification project that predicts the **HS Chapter** of a commodity from its textual description using TF-IDF and machine learning.

---

## 📌 Project Overview

Commodity classification is an important part of international trade and product compliance. Manual classification can require significant time and domain knowledge.

This project explores whether **commodity descriptions can be automatically classified into their corresponding HS Chapters using Natural Language Processing (NLP) and machine learning**.

The project was inspired by my experience in **Product Compliance at Amazon**, where I worked with product classifications for the Canada marketplace.

Since Amazon's internal product and classification data is proprietary, this project uses a publicly available international trade dataset from the **United Nations**.

> **Important:** This project predicts the first two digits of the HS code (HS Chapter). It does not predict the complete Canadian HTS code.

---

## 🎯 Business Problem

Given a commodity description, can a machine learning model predict its corresponding **HS Chapter**?

### Example

**Input:**

```text
Live goats
```

**Predicted HS Chapter:**

```text
01
```

The model learns patterns in commodity descriptions and uses them to classify new descriptions into one of the available HS Chapters.

---

## 📊 Dataset

The project uses the **Global Commodity Trade Statistics** dataset, sourced from United Nations international trade data.

The original dataset contains:

* **8,225,871 trade records**
* **10 columns**

For the NLP task, the following columns were used:

| Column      | Description           |
| ----------- | --------------------- |
| `commodity` | Commodity description |
| `comm_code` | HS/commodity code     |

The HS Chapter was created from the **first two digits of `comm_code`**.

### Example

| HS Code | HS Chapter |
| ------- | ---------- |
| 10410   | 10         |
| 52010   | 52         |
| 87032   | 87         |

---

## 🧹 Data Preparation

The original dataset contains millions of trade records because the same commodity can occur across different countries, years, and trade flows.

The following preprocessing steps were performed:

1. Selected the commodity description and commodity code.
2. Removed aggregate `"ALL COMMODITIES"` records.
3. Removed missing descriptions and codes.
4. Cleaned whitespace from text.
5. Extracted the first two digits of the HS code as the target.
6. Removed duplicate commodity-description/HS Chapter pairs.
7. Identified descriptions associated with multiple HS Chapters.
8. Removed ambiguous descriptions.
9. Retained HS Chapters with at least 6 examples to support reliable stratified 5-fold cross-validation.

### Final Dataset

After preprocessing:

* **5,029 unique commodity-description records**
* **96 HS Chapters**
* Minimum **6 examples per chapter**

---

## 🧠 Machine Learning Approach

This project treats HS Chapter prediction as a **multiclass text classification problem**.

### Target Variable

The target is:

```text
HS Chapter = first 2 digits of the HS code
```

### Input

```text
Commodity description
```

### Output

```text
Predicted HS Chapter
```

---

## 🔤 NLP Feature Engineering — TF-IDF

Commodity descriptions were converted into numerical features using **TF-IDF (Term Frequency–Inverse Document Frequency)**.

The final baseline configuration used:

* Maximum 20,000 features
* Unigrams and bigrams
* English stop-word removal
* Sublinear term-frequency scaling

### Example

For:

```text
cotton woven fabric
```

The model can learn individual words such as:

```text
cotton
woven
fabric
```

and combinations such as:

```text
cotton woven
woven fabric
```

TF-IDF allows the model to represent these textual patterns numerically for machine learning.

---

## 🤖 Models Evaluated

Three baseline models were trained and compared:

### 1. Logistic Regression

A linear classification model used as a baseline for text classification.

### 2. Multinomial Naive Bayes

A probabilistic model commonly used for text classification.

### 3. Linear SVM

A strong linear classifier for high-dimensional TF-IDF text features.

Class weighting was also evaluated to account for differences in the number of examples across HS Chapters.

---

## 📈 Baseline Model Results

The models were evaluated on an **untouched 20% test set**.

| Model                   |   Accuracy |   Macro F1 |   Micro F1 | Weighted F1 |
| ----------------------- | ---------: | ---------: | ---------: | ----------: |
| Logistic Regression     |     62.52% |     41.98% |     62.52% |      60.15% |
| Multinomial Naive Bayes |     45.53% |     20.81% |     45.53% |      42.08% |
| Linear SVM              | **81.01%** | **77.79%** | **81.01%** |  **80.35%** |

Linear SVM performed substantially better than the other baseline models and was therefore selected for further optimization.

---

## ⚙️ Hyperparameter Tuning

The Linear SVM was further optimized using **GridSearchCV with stratified 5-fold cross-validation**.

The tuning process evaluated:

* TF-IDF vocabulary size
* N-gram range
* Sublinear TF
* SVM regularization parameter `C`
* Class weighting

### Search Space

```text
TF-IDF max_features:
10,000 / 20,000

N-gram range:
(1,1) / (1,2)

Sublinear TF:
True / False

SVM C:
0.5 / 1 / 2 / 5

Class weight:
None / balanced
```

This resulted in:

**64 parameter combinations × 5 cross-validation folds = 320 model fits**

The optimization metric was **Macro F1**, because the HS Chapter classes are not evenly represented.

---

## 🏆 Final Model

The best configuration found by GridSearchCV was:

```text
Model: Linear SVM

C: 2

Class Weight: None

TF-IDF Features: 10,000

N-gram Range: (1,1)

Sublinear TF: True
```

### Best Cross-Validation Performance

```text
Macro F1: 77.34%
```

---

## 📊 Final Test Performance

The tuned Linear SVM was evaluated on the test set only after hyperparameter tuning was complete.

### Final Results

| Metric      |      Score |
| ----------- | ---------: |
| Accuracy    | **82.01%** |
| Macro F1    | **79.04%** |
| Micro F1    | **82.01%** |
| Weighted F1 | **81.53%** |

The model correctly classified:

**825 out of 1,006 test samples**

and incorrectly classified:

**181 samples**

### Error Rate

```text
17.99%
```

The **Macro F1 of 79.04%** is particularly useful because it gives equal importance to each HS Chapter rather than allowing larger classes to dominate the metric.

> Note: In single-label multiclass classification, Micro F1 is mathematically equivalent to accuracy.

---

## 🔍 Classification Performance

The final classification report evaluates precision, recall, and F1-score for each HS Chapter.

Some chapters achieve very strong performance, while chapters with fewer examples or descriptions that are linguistically similar to other chapters are more difficult to classify.

This reflects an important characteristic of real-world classification problems: **similar textual descriptions can belong to different categories**, making certain classes inherently more challenging.

---

## 🔎 Error Analysis

The final model produced:

```text
Test samples:        1,006
Incorrect predictions: 181
Error rate:          17.99%
```

Examples of misclassifications included:

| Commodity                                 | Actual | Predicted |
| ----------------------------------------- | -----: | --------: |
| Nitric acid, sulphonitric acids           |     28 |        29 |
| Mate                                      |     09 |        21 |
| Cardamoms                                 |     09 |        29 |
| Honey, natural                            |     04 |        25 |
| X-ray plates and films                    |     37 |        90 |
| Cargo containers designed for carriage    |     86 |        84 |
| Woven fabric polyester + manmade filament |     55 |        54 |

The error analysis helps identify areas where the model struggles, particularly when commodity descriptions contain similar terminology or when a chapter has relatively few training examples.

---

## 📉 Confusion Matrix

A confusion matrix was generated to analyze which HS Chapters are most frequently confused with one another.

This provides a more detailed view of model performance than accuracy alone and helps identify systematic classification errors.

---

## 🧪 Sample Predictions

The trained model was tested on new commodity descriptions:

| Input                      | Prediction |
| -------------------------- | ---------: |
| Live sheep                 |         01 |
| Live goats                 |         01 |
| Cotton woven fabric        |         58 |
| Iron or steel products     |         73 |
| Fresh fish                 |         08 |
| Plastic household articles |         39 |

These examples demonstrate how the trained NLP pipeline can take a commodity description and return an HS Chapter prediction.

---

## 💾 Model Serialization

The final trained pipeline was saved using `joblib`:

```text
hs_chapter_nlp_classifier.pkl
```

The saved pipeline contains:

```text
TF-IDF Vectorizer
        ↓
Tuned Linear SVM
```

This allows the complete preprocessing and prediction pipeline to be loaded without retraining the model.

---

## 🚀 Streamlit Deployment

The trained model is designed to be deployed as an interactive **Streamlit web application**.

The application allows users to:

1. Enter a commodity description.
2. Submit the description.
3. Receive the predicted HS Chapter.

The application uses the saved:

```text
hs_chapter_nlp_classifier.pkl
```

so model training is not required when the application starts.

**Live Demo:** Coming soon

---

## 🗂️ Project Structure

```text
hs-chapter-classification-nlp/
│
├── README.md
├── hs_chapter_classification.ipynb
├── app.py
├── hs_chapter_nlp_classifier.pkl
├── model_comparison_results.csv
└── requirements.txt
```

### File Description

| File                              | Purpose                                            |
| --------------------------------- | -------------------------------------------------- |
| `README.md`                       | Project documentation                              |
| `hs_chapter_classification.ipynb` | Complete data preparation, modeling and evaluation |
| `app.py`                          | Streamlit application                              |
| `hs_chapter_nlp_classifier.pkl`   | Saved trained ML pipeline                          |
| `model_comparison_results.csv`    | Model performance comparison                       |
| `requirements.txt`                | Python dependencies                                |

---

## 🛠️ Technologies Used

### Programming

* Python

### Data Processing

* Pandas
* NumPy

### Machine Learning

* Scikit-learn
* Logistic Regression
* Multinomial Naive Bayes
* Linear SVM
* GridSearchCV
* Stratified Cross-Validation

### NLP

* TF-IDF
* Unigrams
* Bigrams
* Sublinear TF

### Evaluation

* Accuracy
* Macro F1
* Micro F1
* Weighted F1
* Precision
* Recall
* Classification Report
* Confusion Matrix
* Error Analysis

### Visualization

* Matplotlib
* Seaborn

### Deployment

* Joblib
* Streamlit

---

## 💡 Key Learning Outcomes

This project provided practical experience with:

* Working with a large real-world dataset
* Data cleaning and preprocessing
* Converting business problems into ML problems
* NLP text preprocessing
* TF-IDF feature engineering
* Multiclass classification
* Handling class imbalance
* Comparing multiple machine learning algorithms
* Stratified cross-validation
* Hyperparameter tuning
* Selecting appropriate evaluation metrics
* Error analysis
* Model serialization
* Preparing an ML model for deployment

---

## ⚠️ Limitations

This project has several important limitations.

### 1. HS Chapter Rather Than Full HTS Classification

The model predicts the **first two digits of the HS code**, rather than the complete Canadian HTS classification.

### 2. Dataset Characteristics

The dataset contains standardized international commodity descriptions from trade statistics. These descriptions are more structured than typical commercial product titles.

### 3. Limited Examples for Some Chapters

Although chapters with fewer than 6 examples were removed to support reliable cross-validation, some remaining chapters still contain relatively few examples.

### 4. Text-Only Prediction

The model primarily uses the commodity description. It does not use additional product attributes such as:

* Product images
* Brand
* Material specifications
* Country-specific tariff rules
* Detailed product attributes

These additional features could potentially improve classification in a production environment.

---

## 🔮 Future Improvements

Potential future improvements include:

* Expanding the training dataset
* Incorporating richer product attributes
* Exploring domain-specific text preprocessing
* Testing transformer-based NLP models
* Adding model explainability
* Extending prediction from HS Chapter to more detailed HS levels
* Incorporating country-specific tariff rules
* Improving the Streamlit interface

---

## 🎯 Business Relevance

This project demonstrates how an NLP classification system can be used as a **decision-support tool for product classification workflows**.

The model is not intended to replace expert tariff classification. Instead, it demonstrates how machine learning can potentially assist analysts by providing an initial classification that can then be reviewed by a domain expert.

---

## 👩‍💻 Author

**Ayesha Ansari**

Data Science & Data Analytics Aspirant

---

## ⭐ Project Highlights

**Real-world dataset → NLP → Model Comparison → Class Imbalance → Cross-Validation → Hyperparameter Tuning → Error Analysis → Model Serialization → Streamlit Deployment**

---
