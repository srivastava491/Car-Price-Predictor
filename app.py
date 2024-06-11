import pickle
import streamlit as st
import sklearn
import pandas as pd
import numpy
from datetime import datetime

now = datetime.now()

current_year = now.year


cars=pd.read_csv('new_df.csv')
pipe=pickle.load(open("model.pkl","rb"))

brand = st.selectbox(
    "Select Your Car's Brand",
    cars.brand.unique())

model=st.selectbox(
    "Select your Car's Model",
    cars[cars.brand==brand].model.unique())

accident=st.selectbox(
    "How many accidents car has been in",
    cars.accident.unique())

engine=st.selectbox(
    "Select your Car's Engine",
    cars.engine.unique())

fuel=st.selectbox(
    "Select your Car's Model",
    cars.fuel_type.unique())

transmission=st.selectbox(
    "Select your Car's Model",
    cars.transmission.unique())

year = st.number_input("Insert a Model Year",min_value=1980,max_value=current_year,step=1,value=current_year)

milage=st.number_input("Enter Milage",step=1)

if st.button("Predict"):
    data = {
        'brand': [brand],
        'model': [model],
        'fuel_type': [fuel],
        'engine': [engine],
        'milage':[milage],
        'transmission': [transmission],
        'accident': [accident],
        'model_year': [year]
    }
    y_pred=pipe.predict(pd.DataFrame(data))
    st.write(y_pred)
else:
    st.write()