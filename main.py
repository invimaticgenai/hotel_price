import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the pickled model
pickle_file = 'hotel_model.pkl'
with open(pickle_file, 'rb') as f:
    model, scaler = pickle.load(f)

# Streamlit app
st.title("Hotel Price Prediction App")

# User inputs
reviews_count = st.number_input('Reviews Count', min_value=0, value=500, step=1)
rating = st.slider('Rating', min_value=1.0, max_value=10.0, value=8.0, step=0.1)
type_room = st.selectbox('Type of Room', ['Premium single room', 'Premium double room', 'Premium Triple room'])
city = st.selectbox('City', ['Amsterdam', 'Rotterdam', 'Utrecht', 'The Hague'])

# Prepare the input data
type_premium_double_room = 1 if type_room == 'Premium double room' else 0
type_premium_single_room = 1 if type_room == 'Premium single room' else 0
city_rotterdam = 1 if city == 'Rotterdam' else 0
city_the_hague = 1 if city == 'The Hague' else 0
city_utrecht = 1 if city == 'Utrecht' else 0

# Create a DataFrame for the new data point
new_data_point = pd.DataFrame({
    'ReviewsCount': [reviews_count],
    'Rating': [rating],
    'Type_Premium double room': [type_premium_double_room],
    'Type_Premium single room': [type_premium_single_room],
    'City_Rotterdam': [city_rotterdam],
    'City_The Hague': [city_the_hague],
    'City_Utrecht': [city_utrecht]
})

# Ensure the new data point has the same order of columns as the training data
new_data_point = new_data_point[
    ['ReviewsCount', 'Rating', 'Type_Premium double room', 'Type_Premium single room', 'City_Rotterdam',
     'City_The Hague', 'City_Utrecht']]

# When the button is pressed
if st.button('Predict Price'):
    # Standardize the new data point
    new_data_point_scaled = scaler.transform(new_data_point)

    # Predict the price for the new data point
    predicted_price = model.predict(new_data_point_scaled)

    # Display the prediction
    st.write(f"Predicted Price: €{predicted_price[0]:.2f}")
