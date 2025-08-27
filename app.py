import streamlit as st
import pickle
import numpy as np

# Function to load the saved model
@st.cache_data # This caches the model so it doesn't reload every time
def load_model():
    with open('diabetes_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Load the model
model = load_model()

# --- Web App Interface ---

st.title('Diabetes Prediction App 🩺')
st.write("Enter the patient's details below to predict the likelihood of diabetes.")

# Create input fields in two columns for better layout
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, value=6)
    glucose = st.number_input('Glucose', min_value=0, max_value=200, value=148)
    blood_pressure = st.number_input('Blood Pressure (mm Hg)', min_value=0, max_value=140, value=72)
    skin_thickness = st.number_input('Skin Thickness (mm)', min_value=0, max_value=100, value=35)

with col2:
    insulin = st.number_input('Insulin (mu U/ml)', min_value=0, max_value=900, value=0)
    bmi = st.number_input('BMI (weight in kg/(height in m)^2)', min_value=0.0, max_value=70.0, value=33.6, format="%.1f")
    dpf = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.5, value=0.627, format="%.3f")
    age = st.number_input('Age (years)', min_value=0, max_value=120, value=50)

# Create a button to make the prediction
if st.button('*Predict Diabetes*'):
    # Prepare the input data for the model
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    
    # Make prediction
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)
    
    # Display the result
    st.subheader('Prediction Result')
    probability_of_diabetes = prediction_proba[0][1] * 100
    
    if prediction[0] == 1:
        st.error(f'*The model predicts that the patient HAS diabetes.*')
        st.write(f'*Confidence:* {probability_of_diabetes:.2f}%')
    else:
        st.success(f'*The model predicts that the patient DOES NOT have diabetes.*')
        st.write(f'*Confidence:* {100 - probability_of_diabetes:.2f}%')