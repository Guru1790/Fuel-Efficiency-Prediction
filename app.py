import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import joblib
import os

# Load or train the model and scaler
def get_model():
    if os.path.exists('car_mpg_model.h5') and os.path.exists('scaler.pkl'):
        model = keras.models.load_model('car_mpg_model.h5')
        scaler = joblib.load('scaler.pkl')
    else:
        df = load_data()
        (X_train, X_val, Y_train, Y_val), scaler = preprocess_data(df)
        model, scaler, _ = train_model(X_train, Y_train, X_val, Y_val)
    return model, scaler

def load_data():
    df = pd.read_csv('auto-mpg.csv')
    df = df[df['horsepower'] != '?']
    df['horsepower'] = df['horsepower'].astype(float)
    df = df.drop(columns=['car name'])
    df = df.dropna()
    return df

def preprocess_data(df):
    features = df.drop('mpg', axis=1)
    target = df['mpg'].values
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    return train_test_split(features_scaled, target, test_size=0.2, random_state=22), scaler

def build_model(input_shape):
    model = keras.Sequential([
        layers.Dense(256, activation='relu', input_shape=[input_shape]),
        layers.BatchNormalization(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.BatchNormalization(),
        layers.Dense(1, activation='relu')
    ])
    model.compile(
        loss='mae',
        optimizer='adam',
        metrics=['mape']
    )
    return model

def train_model(X_train, Y_train, X_val, Y_val):
    model = build_model(X_train.shape[1])
    
    # Train the model
    history = model.fit(
        X_train, Y_train,
        epochs=50,
        validation_data=(X_val, Y_val),
        verbose=0
    )
    
    # Save the model and scaler
    model.save('car_mpg_model.h5')
    joblib.dump(scaler, 'scaler.pkl')
    
    return model, scaler, history

def predict_mpg(model, scaler, features):
    if scaler is None:
        raise ValueError("Scaler not found. Ensure the model and scaler are loaded properly.")
    features_scaled = scaler.transform([features])  # Transform the features
    prediction = model.predict(features_scaled)
    return prediction[0][0]

# Custom CSS for background image
st.markdown("""
    <style>
        .stApp {
            background-image: url('https://th.bing.com/th/id/R.cc1aae2fb085c350941d2fe89bac1dac?rik=XkW3DOhFgc3eIA&riu=http%3a%2f%2ffile.padamas-forklift.com%2feficiency-e6cb2-290_356.jpg&ehk=tpiht1U59JEZGrRIOZvBQe6BkCbRD5LIz%2fKeFoXJhNw%3d&risl=&pid=ImgRaw&r=0');
            background-size: 90% 90% ;
            background-position: center;
            background-repeat: no-repeat;
        }
    </style>
""", unsafe_allow_html=True)

# Streamlit UI
st.title('Car MPG Prediction')

st.write('Enter the following car features to predict MPG:')

# Input fields for the user
cylinders = st.number_input('Cylinders', min_value=3, max_value=8, value=4)
displacement = st.number_input('Displacement', min_value=70, max_value=500, value=150)
horsepower = st.number_input('Horsepower', min_value=50, max_value=300, value=100)
weight = st.number_input('Weight', min_value=1500, max_value=5000, value=2500)
acceleration = st.number_input('Acceleration', min_value=5, max_value=30, value=15)
origin = st.selectbox('Origin', options=[1, 2, 3], format_func=lambda x: {1: 'USA', 2: 'Europe', 3: 'Japan'}[x])

# Load or train model and scaler
model, scaler = get_model()

# Make predictions
if st.button('Predict'):
    features = [cylinders, displacement, horsepower, weight, acceleration, origin]
    try:
        mpg = predict_mpg(model, scaler, features)
        st.write(f'Predicted MPG: {mpg:.2f}')
    except ValueError as e:
        st.error(f'Error: {e}')
