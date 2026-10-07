# 📦 HS Chapter Classification Using NLP

![Python](https://img.shields.io/badge/Python-3.x-blue) ![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange) ![Streamlit](https://img.shields.io/badge/Streamlit-App-red)

**Commodity Description-Based HS Chapter Classification using TF-IDF and Linear SVM**

Inspired by my experience in Amazon Product Compliance, where I worked on assigning HTS classifications to products for the Canada marketplace, this project explores how Natural Language Processing can be used to classify commodity descriptions into broader HS Chapters.

> **Important:** This project predicts the 2-digit HS Chapter, not the exact Canadian HTS code.

---

## 🚀 Live Demo

👉 **[Try the Streamlit Application](https://hs-chapter-classification-nlp-eazxgagweuzdcaglwnefju.streamlit.app/)**

Enter a commodity description and the application predicts its corresponding HS Chapter using the trained NLP model.

![Streamlit App Screenshot](images/streamlit_app_screenshot.png)

---

## 📌 Project at a Glance

- **Problem:** Classify commodity descriptions into 2-digit HS Chapters using NLP.
- **Data:** 8.2M+ public international trade records, reduced to 5,029 modeling records after data validation and preprocessing.
- **Approach:** TF-IDF text features with Logistic Regression, Multinomial Naive Bayes, and Linear SVM model comparison.
- **Best Model:** Tuned Linear SVM with **82.01% test accuracy** and **79.04% Macro F1**.
- **Deployment:** Interactive Streamlit application for real-time HS Chapter predictions.

---

## 🎯 Business Problem

Product classification is an important part of international trade and product compliance workflows.

During my previous experience in Amazon Product Compliance, I worked with product-level HTS classifications for the Canada marketplace. This motivated me to explore whether commodity/product descriptions could be used to assist classification using machine learning.

Amazon's internal product data is proprietary, so publicly available international trade data was used to build this prototype.

The objective was to:
> Predict the 2-digit HS Chapter from a commodity description using Natural Language Processing.

---

## 💡 Why HS Chapter Instead of Exact HS Code?

During data exploration, I initially considered predicting the exact HS code.

However, after validating the data, I found that exact commodity-code combinations were extremely sparse, with many exact codes having only a single example.

This creates a problem for supervised classification because the model needs multiple examples per class to learn generalizable patterns.

Instead of forcing an unreliable exact-code classifier, I reframed the problem to predict the 2-digit HS Chapter, which provides a broader and better-supported classification target.

This was an important modeling decision based on the structure of the data, not an arbitrary simplification.

---

## 📊 Dataset

The project uses the **Global Commodity Trade Statistics** dataset, sourced from publicly available international trade data.

### Original Dataset
- **Rows:** 8,225,871
- **Columns:** 10

Important columns include:
- `commodity` — commodity description
- `comm_code` — HS commodity code
- `country_or_area`
- `year`
- `flow`
- `trade_usd`
- `weight_kg`
- `quantity`
- `category`

### Data Cleaning
The following steps were performed:
1. Removed `ALL COMMODITIES` aggregate records.
2. Validated missing values.
3. Checked for duplicate records.
4. Examined commodity/code uniqueness.
5. Created the 2-digit HS Chapter target from `comm_code`.
6. Removed records unsuitable for the final NLP formulation.
7. Retained chapters with sufficient examples for reliable cross-validation.

### Final ML Dataset
- **Records:** 5,029
- **HS Chapters:** 96
- **Minimum examples per chapter:** 6

---

## 🧠 Machine Learning Approach

### 1. Text Feature Engineering
Commodity descriptions were converted into numerical features using **TF-IDF (Term Frequency–Inverse Document Frequency)**. TF-IDF helps assign higher importance to terms that are informative for distinguishing between different commodity categories.

### 2. Models Compared
Three baseline models were evaluated:
- Logistic Regression
- Multinomial Naive Bayes
- Linear SVM

### 3. Hyperparameter Tuning
The Linear SVM was selected as the strongest baseline and further optimized using **GridSearchCV with 5-fold cross-validation**.

The search evaluated:
- `C`
- `max_features`
- `ngram_range`
- `sublinear_tf`
- `class_weight`

A total of **64 parameter combinations × 5 folds = 320 model fits** were evaluated.

---

## 📈 Model Performance

| Model | Accuracy | Macro F1 | Micro F1 | Weighted F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 62.52% | 41.98% | 62.52% | 60.15% |
| Multinomial Naive Bayes | 45.53% | 20.81% | 45.53% | 42.08% |
| Linear SVM | 81.01% | 77.79% | 81.01% | 80.35% |
| **Tuned Linear SVM** | **82.01%** | **79.04%** | **82.01%** | **81.53%** |

### Final Model
**Tuned Linear SVM**

Best parameters:
- `C = 2`
- `class_weight = None`
- `max_features = 10,000`
- `ngram_range = (1, 1)`
- `sublinear_tf = True`

### Cross-Validation
Best CV Macro F1: **77.34%**

### Test Set
- Test samples: 1,006
- Incorrect predictions: 181
- Error rate: 17.99%

---

## 🔍 Error Analysis

The final model achieved strong overall performance, but some HS Chapters were more difficult to distinguish.

Examples of incorrect predictions included:

| Commodity | Actual | Predicted |
|---|---:|---:|
| Nitric acid, sulphonitric acids | 28 | 29 |
| Mate | 09 | 21 |
| Furnaces/ovens, non-electric | 84 | 26 |
| Flashbulbs | 90 | 19 |
| Parts of garments | 61 | 62 |
| Cardamoms | 09 | 29 |
| Natural honey | 04 | 25 |
| X-ray plates and films | 37 | 90 |

These errors highlight the difficulty of distinguishing semantically similar or specialized commodity descriptions using relatively short text.

---

## 📉 Confusion Matrix

A confusion matrix was generated to analyze which HS Chapters are most frequently confused with one another, providing a more detailed view of model performance than accuracy alone.

![Confusion Matrix](images/confusion_matrix.png)

---

## 📊 Model Evaluation

The project evaluates the model using:
- Accuracy
- Macro F1
- Micro F1
- Weighted F1
- Classification report
- Confusion matrix
- Error analysis

Macro F1 was particularly important because the dataset contains classes with different levels of support.

---

## 🖥️ Streamlit Deployment

The trained pipeline was serialized using `joblib` and deployed as an interactive Streamlit application.

The application:
1. Loads the trained TF-IDF + Linear SVM pipeline.
2. Accepts a commodity description from the user.
3. Applies the same TF-IDF transformation used during training.
4. Predicts the corresponding HS Chapter.
5. Displays the prediction in the web interface.

**Example**

Input:
```text
Live sheep
```
Expected prediction:
```text
HS Chapter 01
```

👉 **[Launch the Streamlit App](REPLACE_WITH_YOUR_STREAMLIT_APP_LINK)**

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TF-IDF
- Logistic Regression
- Multinomial Naive Bayes
- Linear SVM
- GridSearchCV
- Joblib
- Streamlit
- Google Colab
- GitHub

---

## 📁 Project Structure

```text
hs-chapter-classification-nlp/
│
├── LICENSE
├── README.md
├── app.py
├── hs_chapter_classification.ipynb
├── hs_chapter_nlp_classifier.pkl
├── model_comparison_results.csv
├── requirements.txt
└── images/
    ├── confusion_matrix.png
    └── streamlit_app_screenshot.png
```

### File Description

| File | Description |
|---|---|
| `README.md` | Project documentation, methodology, results, and deployment details |
| `app.py` | Streamlit application for HS Chapter prediction |
| `hs_chapter_classification.ipynb` | Complete data preprocessing, model training, evaluation, and tuning workflow |
| `hs_chapter_nlp_classifier.pkl` | Saved trained TF-IDF + Linear SVM pipeline |
| `model_comparison_results.csv` | Model performance comparison results |
| `requirements.txt` | Python dependencies required to run the Streamlit application |
| `LICENSE` | Project license |

---

## 💾 Model Serialization

The final trained pipeline was saved as:
```text
hs_chapter_nlp_classifier.pkl
```
The saved pipeline contains the text preprocessing and trained classifier, allowing the Streamlit application to make predictions without retraining the model.

---

## 🎓 Key Learning Outcomes

This project helped me strengthen my understanding of:
- NLP text classification
- TF-IDF feature engineering
- Multiclass classification
- Linear SVM
- Model comparison
- Hyperparameter tuning
- Cross-validation
- Class imbalance
- Macro vs. Weighted F1
- Error analysis
- Model serialization
- Streamlit deployment
- GitHub project organization

Most importantly, the project reinforced the importance of problem formulation and data validation before model selection.

---

## ⚠️ Limitations

- The model predicts the 2-digit HS Chapter, not the exact Canadian HTS classification.
- The dataset contains standardized international trade commodity descriptions rather than real Amazon product titles.
- Some HS Chapters have relatively low support.
- The model achieves 82.01% accuracy, meaning incorrect predictions still occur.
- The model should be treated as a classification prototype rather than an automated compliance decision system.

---

## 🔮 Future Improvements

- Collecting more product-level labeled descriptions.
- Testing character-level TF-IDF features.
- Using word and character n-gram combinations.
- Exploring hierarchical HS classification.
- Testing transformer-based NLP models.
- Improving validation using genuinely unseen commodity descriptions.
- Adding top-k predictions for analyst review.
- Adding confidence/calibration techniques appropriate for the final model.
- Integrating human-in-the-loop review for compliance workflows.

---

## 💼 Business Relevance

A production-grade version of this concept could potentially support product classification workflows by providing a first-pass recommendation to a human reviewer.

Instead of replacing human compliance decisions, the system could support this flow:

**Product Description → ML Recommendation → Human Review → Final Classification**

This could help prioritize analyst attention and reduce repetitive classification work while maintaining human oversight.

---

## 👩‍💻 Author

**Ayesha Ansari**
Data Science & Data Analytics Aspirant
GitHub: [https://github.com/Ayesha-Ansa](https://github.com/Ayesha-Ansa)

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
