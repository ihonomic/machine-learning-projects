import pickle

import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu

# Load models
diabetes_model = pickle.load(open("diabetes_model.sav", "rb"))
breastcancer_model = pickle.load(open("breastcancer_model.sav", "rb"))
parkinson_model = pickle.load(open("parkinson_model.sav", "rb"))


def diabetes_prediction(input_data):
    input_data = np.asarray(input_data)
    input_data = input_data.reshape(1, -1)
    prediction = diabetes_model.predict(input_data)

    # print(prediction)

    if prediction[0] == 0:
        return "Non-Diabetic"
    else:
        return "Diabetic"


def breastcancer_prediction(input_data):
    input_data = [float(i) for i in input_data]
    input_data = np.asarray(input_data)
    input_data = input_data.reshape(1, -1)

    prediction = breastcancer_model.predict(input_data)

    # print(prediction)

    if prediction[0] == "B":
        return "This values reads: Benign"
    else:
        return "This values reads: Malignant"


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

    col1, col2, col3 = st.columns(3)
    with col1:
        radius_mean = st.text_input("radius_mean")
    with col2:
        texture_mean = st.text_input("texture_mean")
    with col3:
        perimeter_mean = st.text_input("perimeter_mean")
    with col1:
        area_mean = st.text_input("area_mean")
    with col2:
        smoothness_mean = st.text_input("smoothness_mean")
    with col3:
        compactness_mean = st.text_input("compactness_mean")
    with col1:
        concavity_mean = st.text_input("concavity_mean")
    with col2:
        concave_points_mean = st.text_input("concave_points_mean")
    with col3:
        symmetry_mean = st.text_input("symmetry_mean")
    with col1:
        fractal_dimension_mean = st.text_input("fractal_dimension_mean")
    with col2:
        radius_se = st.text_input("radius_se")
    with col3:
        texture_se = st.text_input("texture_se")
    with col1:
        perimeter_se = st.text_input("perimeter_se")
    with col2:
        area_se = st.text_input("area_se")
    with col3:
        smoothness_se = st.text_input("smoothness_se")
    with col1:
        compactness_se = st.text_input("compactness_se")
    with col2:
        concavity_se = st.text_input("concavity_se")
    with col3:
        concave_points_se = st.text_input("concave_points_se")
    with col1:
        symmetry_se = st.text_input("symmetry_se")
    with col2:
        fractal_dimension_se = st.text_input("fractal_dimension_se")
    with col3:
        radius_worst = st.text_input("radius_worst")
    with col1:
        texture_worst = st.text_input("texture_worst")
    with col2:
        perimeter_worst = st.text_input("perimeter_worst")
    with col3:
        area_worst = st.text_input("area_worst")
    with col1:
        smoothness_worst = st.text_input("smoothness_worst")
    with col2:
        compactness_worst = st.text_input("compactness_worst")
    with col3:
        concavity_worst = st.text_input("concavity_worst")
    with col1:
        concave_points_worst = st.text_input("concave_points_worst")
    with col2:
        symmetry_worst = st.text_input("symmetry_worst")
    with col3:
        fractal_dimension_worst = st.text_input("fractal_dimension_worst")

    diagnosis = ""

    # Button
    if st.button("Breast cancer Test Result"):
        diagnosis = breastcancer_prediction(
            [
                radius_mean,
                texture_mean,
                perimeter_mean,
                area_mean,
                smoothness_mean,
                compactness_mean,
                concavity_mean,
                concave_points_mean,
                symmetry_mean,
                fractal_dimension_mean,
                radius_se,
                texture_se,
                perimeter_se,
                area_se,
                smoothness_se,
                compactness_se,
                concavity_se,
                concave_points_se,
                symmetry_se,
                fractal_dimension_se,
                radius_worst,
                texture_worst,
                perimeter_worst,
                area_worst,
                smoothness_worst,
                compactness_worst,
                concavity_worst,
                concave_points_worst,
                symmetry_worst,
                fractal_dimension_worst,
            ]
        )

    st.success(diagnosis)

else:
    st.title("Parkinson Prediction using ML")
