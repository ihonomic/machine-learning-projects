import pickle

import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler

loaded_model = pickle.load(open("diabetes_model.sav", "rb"))
scaler = StandardScaler()


def diabetes_prediction(input_data):
    input_data = (13, 145, 82, 19, 110, 22.2, 0.245, 57)

    # Change the input data to numpy array
    input_data = np.asarray(input_data)

    # Reshape the array for the model to understand since we're predicting for one instace
    input_data = input_data.reshape(1, -1)
    input_data = scaler.fit_transform(input_data)

    prediction = loaded_model.predict(input_data)

    print(prediction)

    if prediction[0] == 0:
        return "Non-Diabetic"
    else:
        return "Diabetic"


def main():
    # Title
    st.title("Diabetes Prediction Web App")

    # Feature Input from the user
    Pregnancies = st.text_input("Number of pregnancies: ")
    Glucose = st.text_input("Glucose Level: ")
    BloodPressure = st.text_input("Blood Pressure value: ")
    SkinThickness = st.text_input("Skin thickness: ")
    Insulin = st.text_input("Insulin level: ")
    BMI = st.text_input("BMI (Body mass) value: ")
    DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree function: ")
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
