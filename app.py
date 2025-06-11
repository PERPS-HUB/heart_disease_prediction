import streamlit as st
import numpy as np
import joblib

#Load the pre-trained model
model = joblib.load('svm_model.pkl')
scaler = joblib.load('scaler.pkl')

feature_name=['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']

st.title('Heart Disease Prediction')

#Input fields for user data
user_input= []
for feature in feature_name:
    val=st.number_input(f"Enter value for {feature}:", min_value=0.0, step=0.5)
    user_input.append(val)

if st.button('predict'):
    X=np.array(user_input).reshape(1, -1)
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)
    st.success(f"Prediction: {prediction[0]}")

if prediction[0] == 1:
    st.warning("You may have heart disease. Please consult a doctor.")
else:
    st.success("You are likely to be healthy. Keep up the good work!")