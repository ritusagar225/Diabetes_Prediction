# 🩺 Diabetes Prediction using Machine Learning


👉 **[🔗 Live Demo](https://ritusagar225-diabetes-prediction-app-vyknrv.streamlit.app/)**

This project aims to predict whether a person has diabetes based on diagnostic measurements. It uses machine learning algorithms trained on the **PIMA Indians Diabetes Dataset** to provide early prediction and assist in proactive healthcare planning.

---

## 📌 Problem Statement

The growing prevalence of diabetes poses a serious threat to global health. Early detection is essential to prevent long-term complications. However, many remain undiagnosed due to lack of awareness and access to medical facilities. This project uses ML techniques to predict diabetes using readily available health features.

---

## 🎯 Objective

- Build a machine learning model to accurately classify individuals as diabetic or non-diabetic.
- Evaluate performance across different ML algorithms.
- Provide an interactive interface for users to input data and get predictions.
- Aid healthcare professionals in decision-making with data-driven insights.

---

## 🧠 Algorithms Used

- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest Classifier ✅
- K-Nearest Neighbors (KNN)

The final deployed model uses **Random Forest** due to its higher accuracy and robustness.

---

## 📂 Dataset

- **Source:** UCI Machine Learning Repository
- **Features:**
  - Pregnancies
  - Glucose
  - BloodPressure
  - SkinThickness
  - Insulin
  - BMI
  - DiabetesPedigreeFunction
  - Age
- **Target:**
  - Outcome (0: Non-Diabetic, 1: Diabetic)

---

## 🛠️ Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Joblib (for model saving)
- Streamlit (for web app deployment)

---

## 🧪 Evaluation Metrics

- Accuracy Score
- Precision, Recall, F1-score
- Confusion Matrix

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/diabetes-prediction.git
   cd diabetes-prediction

2. Install Dependencies:
   ```bash
   pip install -r requirements.txt

4. Run the app:
   ```bash
   streamlit run app.py

