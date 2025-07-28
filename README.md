
# 💳 Credit Card Fraud Detection

This project focuses on detecting fraudulent credit card transactions using machine learning models. We use the popular Kaggle credit card fraud dataset and apply multiple classification algorithms to identify potentially fraudulent behavior.

## 📁 Dataset

- Source: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- Features:
  - 284,807 transactions
  - 492 frauds (highly imbalanced dataset)
  - PCA-transformed features (V1 to V28), plus `Time`, `Amount`, and `Class` (0 = legit, 1 = fraud)

---

## 🧠 Algorithms Used

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

---

## ⚙️ Workflow

1. **Data Preprocessing**
   - Standardize `Amount` and `Time`
   - Handle class imbalance using `RandomUnderSampler`

2. **Model Training**
   - Train/test split (80/20)
   - Fit each model on the resampled data

3. **Evaluation Metrics**
   - Confusion Matrix
   - Precision, Recall, F1-score
   - ROC-AUC Score
   - ROC Curve visualization

---

## 📊 Results Summary

| Model               | ROC AUC Score | Accuracy | Precision | Recall |
|--------------------|---------------|----------|-----------|--------|
| Logistic Regression| 0.97          | 92%      | 0.96      | 0.89   |
| Decision Tree      | 0.87          | 87%      | 0.90      | 0.84   |
| Random Forest      | **0.98**      | 92%      | 0.96      | 0.89   |

✅ **Random Forest** performed best in terms of AUC and overall balance.

---

## 📈 ROC Curve


<img width="797" height="680" alt="Screenshot 2025-07-28 104427" src="https://github.com/user-attachments/assets/8428532f-56c4-46d8-b800-72409afb18a8" />

---

## 📦 Installation

```bash
pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn
