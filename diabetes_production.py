import pickle

import numpy as np
import streamlit as st


def diabetes_prediction(input_data):

    loaded_model = pickle.load(open("diabetes_model.sav", "rb"))

    # Change the input data to numpy array
    input_data = np.asarray(input_data)

    # Reshape the array for the model to understand since we're predicting for one instace
    input_data = input_data.reshape(1, -1)

    prediction = loaded_model.predict(input_data)

    print(prediction)

    if prediction[0] == 0:
        return "Non-Diabetic"
    else:
        return "Diabetic"


def main():
    # Title
    st.title("Diabetes Prediction Web App")

    # Columns
    col1, col2, col3 = st.columns(3)

    # Feature Input from the user
    with col1:
        Pregnancies = st.text_input("Number of pregnancies: ")
    with col2:
        Glucose = st.text_input("Glucose Level: ")
    with col3:
        BloodPressure = st.text_input("Blood Pressure value: ")
    with col1:
        SkinThickness = st.text_input("Skin thickness: ")
    with col2:
        Insulin = st.text_input("Insulin level: ")
    with col3:
        BMI = st.text_input("BMI (Body mass) value: ")
    with col1:
        DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree function: ")
    with col2:
        Age = st.text_input("Age of person: ")

    # Prediction
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


if __name__ == "__main__":
    main()
