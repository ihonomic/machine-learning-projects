import pickle

import numpy as np

filename = "trained_diabetic_model.sav"

loaded_model = pickle.load(open(filename, "rb"))

input_data = (13, 145, 82, 19, 110, 22.2, 0.245, 57)

# Change the input data to numpy array
input_data = np.asarray(input_data)

# Reshape the array for the model to understand since we're predicting for one instace
input_data = input_data.reshape(1, -1)

# Standardized data
# input_data = scaler.transform(input_data)

prediction = loaded_model.predict(input_data)

print(prediction)

if prediction[0] == 0:
    print("Non-Diabetic")
else:
    print("Diabetic")
