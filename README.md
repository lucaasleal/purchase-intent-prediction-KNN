# Purchase Intent Prediction — KNN Experiment

This project implements a Machine Learning pipeline to predict **purchase intent**
of online shoppers based on session behavior data.  
The core focus is not only classification accuracy, but **experimental analysis of how
the K-Nearest Neighbors (KNN) hyperparameter `k` affects model behavior**, especially
the trade-off between sensitivity and specificity.

---

## 🧠 Project Idea

The goal is to predict whether a user session will result in a purchase
(`Revenue = TRUE`) using historical browsing and interaction data from an
e-commerce platform.

Instead of treating the model as a black box, this project **systematically compares
different values of `k` in the KNN algorithm**, highlighting how model behavior
changes as neighborhood size increases.

Key questions explored:

- How does increasing `k` affect false positives and false negatives?
- Is higher accuracy always the best metric?
- How does the bias–variance trade-off manifest in KNN?

---

## 📊 Data Modeling

The dataset contains real user session data composed of numerical and categorical
features.

### Features (Evidence)

- Administrative (int) and _Duration (float)
- Informational (int) and _Duration (float)
- ProductRelated (int) and _Duration (float)
- BounceRates, ExitRate, PageValues, SpecialDay (float)
- Month (categorical, encoded as 0–11)
- OperatingSystems, Browser, Region, TrafficType (int)
- VisitorType (binary: Returning / New)
- Weekend (binary)

### Label

- Revenue  
  - `1` → Purchase  
  - `0` → No purchase

---

## ⚙️ Technologies Used

- Python 3
- Scikit-learn
- K-Nearest Neighbors (KNN)
- Train/Test Split
- Evaluation Metrics:
  - Accuracy
  - Sensitivity (True Positive Rate)
  - Specificity (True Negative Rate)

---

## 🧪 Experimental Setup

The experiment compares **three KNN models** with different neighborhood sizes:

- `k = 1`
- `k = 3`
- `k = 5`

Each model is trained and evaluated on the same train/test split to ensure
a fair comparison.

---

## 📈 Results

| K | Accuracy | Sensitivity (TPR) | Specificity (TNR) |
|---|----------|------------------|------------------|
| 1 | 83.09%   | 38.94%           | 90.85%           |
| 3 | 85.12%   | 33.78%           | 94.22%           |
| 5 | 86.05%   | 28.82%           | 96.25%           |

---

## 🧠 Analysis

As the value of `k` increases, the model becomes more conservative.

- **Accuracy increases** steadily with larger `k`
- **Specificity improves significantly**, meaning fewer false positives
- **Sensitivity drops sharply**, indicating that many true buyers are missed

This clearly demonstrates the **bias–variance trade-off** inherent in KNN models:

- Small `k` → High variance, more aggressive predictions
- Large `k` → High bias, conservative behavior

The optimal choice of `k` depends on the business objective:
- Marketing-focused strategies may prefer higher sensitivity
- Cost-sensitive strategies may prefer higher specificity

### Confusion Matrix (K = 5)
<img width="640" height="480" alt="confusion_matrix_k5" src="https://github.com/user-attachments/assets/97f3e5f3-94e6-447a-a0c7-502a2015aae1" />

---

## ▶️ How to Run

Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/lucaasleal/purchase-intent-prediction-KNN.git
cd purchase-intent-prediction-KNN
```

Install dependencies:
```bash
pip install -r requirements.txt
```
Run (Select your K in train_model function):
```bash
python src/main.py
```

## 🚀 Possible Improvements

Apply feature normalization (StandardScaler) and re-evaluate KNN

Compare KNN with other classifiers (Logistic Regression, Decision Trees)

Perform cross-validation for more robust results

## 📚 Dataset Source
UCI Machine Learning Repository — Online Shoppers Purchasing Intention Dataset

## 👤 Author
@lucaasleal
