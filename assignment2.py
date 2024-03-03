import streamlit as st
import pandas as pd
import joblib
import os 

current_dir = os.path.dirname(__file__)
model_path = os.path.join(current_dir, 'model.sav')
st.title('Multiple Linear Regression Model Prediction')
model = joblib.load(model_path)

def user_input():
    BMI = st.sidebar.slider('BMI', 0, 80, 25)
    data = {'BMXBMI': BMI}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input()

st.subheader("User input")
st.write(df)

if st.button('Predict'):
    # Make sure the features are in the same order as during model training
    prediction = model.predict(df)
    st.write(f'Predicted outcome: {prediction[0]}')



