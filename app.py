import streamlit as st 
import pickle
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
features = X.columns

f = open('model.pkl', 'rb')
model = pickle.load(f)

st.title("Breast Cancer Prediction")
st.write("Enter Feature Values :")

input_data = []

for feature in features:
    value = st.number_input(f"{feature}", value=0.0)
    input_data.append(value)

if st.button("Predict"):
    input_array = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_array)


    if (prediction):
        st.warning
    else:
        st.success