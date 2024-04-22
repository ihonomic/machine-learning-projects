import pickle

import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu

# Load models
diabetes_model = pickle.load(open("diabetes_model.sav", "rb"))
breastcancer_model = pickle.load(open("breastcancer_model.sav", "rb"))
parkinson_model = pickle.load(open("parkinson_model.sav", "rb"))


def diabetes_prediction(input_data):

    # Change the input data to numpy array
    input_data = np.asarray(input_data)

    # Reshape the array for the model to understand since we're predicting for one instace
    input_data = input_data.reshape(1, -1)

    prediction = diabetes_model.predict(input_data)

    print(prediction)

    if prediction[0] == 0:
        return "Non-Diabetic"
    else:
        return "Diabetic"


# Sidebar for navigation
with st.sidebar:
    selected = option_menu(
        "Multiple Disease Prediction system",
        ["Diabetes Prediction", "Breast Cancer Prediction", "Parkinson Prediction"],
        icons=["activity", "heart", "person"],  # bootstrap icons
        default_index=0,
    )

if selected == "Diabetes Prediction":
    st.title("Diabetes Prediction using ML")

    Pregnancies = st.text_input("Number of pregnancies: ")
    Glucose = st.text_input("Glucose Level: ")
    BloodPressure = st.text_input("Blood Pressure value: ")
    SkinThickness = st.text_input("Skin thickness: ")
    Insulin = st.text_input("Insulin level: ")
    BMI = st.text_input("BMI (Body mass) value: ")
    DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree function: ")
    Age = st.text_input("Age of person: ")

    diagnosis = ""

    # Button
    if st.button("Diabetes Test Result"):
        diagnosis = diabetes_prediction(
            [
                Pregnancies,
                Glucose,
                BloodPressure,
                SkinThickness,
                Insulin,
                BMI,
                DiabetesPedigreeFunction,
                Age,
            ]
        )

    st.success(diagnosis)

elif selected == "Breast Cancer Prediction":
    st.title("Breast Cancer Prediction using ML")

else:
    st.title("Parkinson Prediction using ML")
