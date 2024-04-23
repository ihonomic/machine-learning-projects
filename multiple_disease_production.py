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


def parkinson_prediction(input_data):
    input_data = np.asarray(input_data)
    input_data = input_data.reshape(1, -1)

    prediction = parkinson_model.predict(input_data)

    # print(prediction)

    if prediction[0] == 1:
        return "This values reads: Positive to parkinson disease"
    else:
        return "This values reads: Negative to parkinson disease"


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
    st.text("Model Trained using Support vector Machine SVM")

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
    st.text("Model Trained using Logistic Regression")

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
    st.text("Model Trained using Support vector Machine SVM")

    col1, col2, col3 = st.columns(3)
    with col1:
        MDVP_Fo = st.text_input("MDVP_Fo(Hz)")
    with col2:
        MDVP_Fhi = st.text_input("MDVP_Fhi(Hz)")
    with col3:
        MDVP_Flo = st.text_input("MDVP_Flo(Hz)")
    with col1:
        MDVP_jitter = st.text_input("MDVP_Jitter(%)")
    with col2:
        MDVP_jitter_abs = st.text_input("MDVP_Jitter(Abs)")
    with col3:
        MDVP_rap = st.text_input("MDVP_RAP")
    with col1:
        MDVP_ppq = st.text_input("MDVP_PPQ")
    with col2:
        jitter_ddp = st.text_input("Jitter_DDP")
    with col3:
        MDVP_shimmer = st.text_input("MDVP_Shimmer")
    with col1:
        MDVP_shimmer_db = st.text_input("MDVP_Shimmer(dB)")
    with col2:
        shimmer_apq3 = st.text_input("Shimmer_APQ3")
    with col3:
        shimmer_apq5 = st.text_input("Shimmer_APQ5")
    with col1:
        MDVP_apq = st.text_input("MDVP_APQ")
    with col2:
        shimmer_dda = st.text_input("Shimmer_DDA")
    with col3:
        nhr = st.text_input("NHR")
    with col1:
        hnr = st.text_input("HNR")
    # with col2:
    #     status = st.text_input("status")  #
    with col2:
        rpde = st.text_input("RPDE")
    with col3:
        dfa = st.text_input("DFA")
    with col1:
        spread1 = st.text_input("spread1")
    with col2:
        spread2 = st.text_input("spread2")
    with col3:
        d2 = st.text_input("D2")
    with col1:
        ppe = st.text_input("PPE")

    diagnosis = ""

    # Button
    if st.button("Parkinson Test Result"):
        diagnosis = parkinson_prediction(
            [
                MDVP_Fo,
                MDVP_Fhi,
                MDVP_Flo,
                MDVP_jitter,
                MDVP_jitter_abs,
                MDVP_rap,
                MDVP_ppq,
                jitter_ddp,
                MDVP_shimmer,
                MDVP_shimmer_db,
                shimmer_apq3,
                shimmer_apq5,
                MDVP_apq,
                shimmer_dda,
                nhr,
                hnr,
                # status,
                rpde,
                dfa,
                spread1,
                spread2,
                d2,
                ppe,
            ]
        )

    st.success(diagnosis)
