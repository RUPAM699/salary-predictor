import streamlit as st
import numpy as np
import pickle

# Load model and preprocessing tools
with open("GradientBoostingClassifier_best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

st.title("Adult Income Prediction App")
st.write("Predict whether a person earns more than 50K or not.")

# Input fields
age = st.number_input("Age", 18, 90, 30)
workclass = st.selectbox("Workclass", ['Private', 'Self-emp-not-inc', 'Local-gov', 'State-gov', 'Federal-gov', 'Others'])

marital_status = st.selectbox("Marital Status", ['Never-married', 'Married-civ-spouse', 'Divorced', 'Separated', 'Widowed', 'Married-spouse-absent', 'Others'])
occupation = st.selectbox("Occupation", ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Others'])
relationship = st.selectbox("Relationship", ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'])
race = st.selectbox("Race", ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'])
sex = st.selectbox("Sex", ['Male', 'Female'])
hours_per_week = st.slider("Hours per Week", 1, 100, 40)
native_country = st.selectbox("Native Country", ['United-States', 'Mexico', 'Philippines', 'Germany', 'Canada', 'India', 'Others'])

# Create input array
input_data = {
    'age': age,
    'workclass': workclass,
    'marital-status': marital_status,
    'occupation': occupation,
    'relationship': relationship,
    'race': race,
    'sex': sex,
    'hours-per-week': hours_per_week,
    'native-country': native_country
}

# Encoding categorical features
for col in label_encoders:
    if col in input_data:
        encoder = label_encoders[col]
        try:
            input_data[col] = encoder.transform([input_data[col]])[0]
        except ValueError:
            input_data[col] = encoder.transform(['Others'])[0]

# Convert to array and scale
input_array = np.array([list(input_data.values())])
input_scaled = scaler.transform(input_array)

# Predict
if st.button("Predict"):
    pred = model.predict(input_scaled)
    income = ">50K" if pred[0] == 1 else "<=50K"
    st.success(f"Predicted Income: {income}")
