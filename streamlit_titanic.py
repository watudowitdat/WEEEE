import streamlit as st
import pandas as pd
import joblib
import os


script_dir = os.path.dirname(os.path.abspath(__file__))
my_data_path = os.path.join(script_dir, "titanic_input_data.csv")
my_model_path = os.path.join(script_dir, "model.pkl")

# Verify the correct loads
print(my_data_path)
print(my_model_path)

titanic_input_data = pd.read_csv(my_data_path)

print(titanic_input_data.head())

# Load model
model = joblib.load(my_model_path)

# Make predictions
x = model.predict(titanic_input_data)

print(x)

st.write('Hi There')
pclass = st.number_input('Enter passenger class, 1-3', min_value=1, max_value=3)
sex_female = st.number_input('Enter 1 if female, 0 if male', min_value=0, max_value=1)
sex_male = st.number_input('Enter 1 if male, 0 if female', min_value=0, max_value=1)
age = st.number_input('Enter age, 1-100', min_value=1, max_value=100)
fare = st.number_input('Enter fare, 0-500')

input_seq = [[pclass, sex_female, sex_male, age, fare]]

x = model.predict(input_seq)

if st.button("Predict"):
    # Create input DataFrame
    input_df = pd.DataFrame({
        "Pclass": [pclass],
        "Sex_female": [sex_female],
        "Sex_male": [sex_male],
        "Age": [age],
        "Fare": [fare]

    })

    # Predict
    prediction = model.predict(input_df)[0]

    # Display result
    result = "Survived" if prediction == 1 else "Did not survive"
    st.success(f"Prediction: {result}")

