# 💼 Adult Income Prediction App (WageWizard)

This project aims to predict whether an individual's annual income exceeds $50,000 using demographic and employment-related features. Built with machine learning and deployed via Streamlit, the app allows users to interactively input data and receive real-time predictions.

---

## 🔍 Problem Statement

Many organizations and researchers are interested in identifying income groups based on socio-economic attributes. This project addresses the challenge of predicting whether a person earns more than $50,000 annually using the UCI Adult dataset. The goal is to create a machine learning model that can accurately classify individuals and deploy it as a user-friendly web application.

---

## 📂 Project Structure
```
├── salary_predictor.py # Main Streamlit application
├── GradientBoostingClassifier_best_model.pkl (Gradient Boosting Classifier)
├── model_training.ipynb ( training model )
├── adult.csv # Dataset used for training (optional upload)

```

---

## 🧰 System Requirements

- Python 3.8+
- RAM: 4GB minimum (8GB recommended)
- Streamlit-compatible browser (Chrome, Firefox, etc.)

---

## 📚 Libraries Used

- `pandas` – Data manipulation  
- `numpy` – Numerical computations  
- `scikit-learn` – Machine learning models and preprocessing  
- `matplotlib`, `seaborn` – Data visualization  
- `pickle`, `joblib` – Model saving/loading  
- `streamlit` – Web app deployment  

Install all dependencies using:

```bash
pip install -r requirements.txt
```

