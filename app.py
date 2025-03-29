import streamlit as st
import numpy as np
import pickle  # To load the model

# Load the trained XGBoost model
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

st.title("Health Prediction using XGBoost")

# Example symptoms (Modify as per your dataset)
symptom_list = ["Fever", "Cough", "Headache", "Fatigue"]
selected_symptoms = []

# Create checkboxes for symptoms
for symptom in symptom_list:
    if st.checkbox(symptom):
        selected_symptoms.append(1)
    else:
        selected_symptoms.append(0)

def predict_disease(user_input):
    prediction = model.predict(user_input)  # Predict using XGBoost model
    return prediction[0]  # Return the predicted class

if st.button("Predict"):
    user_input = np.array(selected_symptoms).reshape(1, -1)
    disease = predict_disease(user_input)  
    st.write(f"Predicted Disease: {disease}")


