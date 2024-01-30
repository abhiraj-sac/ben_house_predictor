import pickle

import streamlit as st
import pandas as pd
import numpy as np

# Assume 'your_model' is a trained machine learning model
# You need to replace this with your actual model
# For demonstration purposes, I'm using a simple linear regression model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import precision_score

pipe = pickle.load(open('Ridgereg.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

st.title("laptop Predictor")
area_type = st.selectbox('area_type', df['area_type'].unique())
availability = st.selectbox('availability', df['availability'].unique())
location = st.selectbox('location', df['location'].unique())
total_sqft = st.number_input('total_sqft')
bath = st.selectbox('bath', df['bath'].unique())
bhk = st.selectbox('bhk', df['bhk'].unique())



if st.button('Predict Price'):
    pred = pipe.predict([[area_type,availability,location,total_sqft,bath,bhk]])
    st.header(pred * 1e5)
