import json
import pickle

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class ModelInput(BaseModel):
    Pregnancies: int
    Glucose: int
    BloodPressure: int
    SkinThickness: int
    Insulin: int
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int


# Load the model
diabetes_model = pickle.load(open("diabetes_model.sav", "rb"))


@app.post("/diabetes_prediction")
def diabetes_pred(input_parameters: ModelInput):
    input_data = input_parameters.model_dump_json()
    input_dic = json.loads(input_data)

    pregnancy = input_dic["Pregnancies"]
    glucose = input_dic["Glucose"]
    blood_pressure = input_dic["BloodPressure"]
    skin_thickness = input_dic["SkinThickness"]
    insulin = input_dic["Insulin"]
    bmi = input_dic["BMI"]
    dpf = input_dic["DiabetesPedigreeFunction"]
    age = input_dic["Age"]

    data = [pregnancy, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]

    prediction = diabetes_model.predict([data])

    if prediction[0] == 0:
        return "Non-diabetic"
    return "Diabetic"
