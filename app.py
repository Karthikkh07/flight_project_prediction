import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(
    page_title="✈️ Flight Price Predictor",
    page_icon="✈️",
    layout="wide"
)

# -------------------------------------------------
# Load trained ML model
# -------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("flight_price_model.pkl")

model = load_model()

# -------------------------------------------------
# Custom CSS
# -------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1504198266280-5c4f47d3b3df");
        background-size: cover;
        background-attachment: fixed;
    }
    h1, h2, h3 {
        color: #caf0f8;
        text-shadow: 1px 1px 2px black;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------
# Sidebar Inputs
# -------------------------------------------------
st.sidebar.title("✈️ Enter Flight Details")

airline = st.sidebar.selectbox(
    "Airline",
    ["IndiGo", "Air India", "SpiceJet", "GoAir", "Vistara"]
)

source = st.sidebar.selectbox(
    "Source",
    ["Delhi", "Kolkata", "Mumbai", "Chennai", "Bangalore"]
)

destination = st.sidebar.selectbox(
    "Destination",
    ["Cochin", "Delhi", "New Delhi", "Hyderabad", "Kolkata"]
)

total_stops = st.sidebar.selectbox("Total Stops", [0, 1, 2, 3, 4])
journey_date = st.sidebar.date_input("Date of Journey")
dep_time = st.sidebar.time_input("Departure Time")
arr_time = st.sidebar.time_input("Arrival Time")

route = st.sidebar.text_input("Route", "NA")
additional_info = st.sidebar.text_input("Additional Info", "No info")

# -------------------------------------------------
# Feature Engineering Function
# -------------------------------------------------
def prepare_input():
    journey_day = journey_date.day
    journey_month = journey_date.month

    dep_hour = dep_time.hour
    dep_min = dep_time.minute

    arr_hour = arr_time.hour
    arr_min = arr_time.minute

    duration_mins = (arr_hour * 60 + arr_min) - (dep_hour * 60 + dep_min)
    if duration_mins < 0:
        duration_mins += 24 * 60

    data = {
        "Airline": airline,
        "Source": source,
        "Destination": destination,
        "Route": route,
        "Additional_Info": additional_info,
        "Total_Stops": str(total_stops),
        "Journey_Day": journey_day,
        "Journey_Month": journey_month,
        "Dep_Hour": dep_hour,
        "Dep_Min": dep_min,
        "Arr_Hour": arr_hour,
        "Arr_Min": arr_min,
        "Duration_mins": duration_mins,
        "Total_Stops_Num": total_stops
    }

    return pd.DataFrame([data])

# -------------------------------------------------
# Main UI
# -------------------------------------------------
st.markdown(
    """
    <h1>✈️ Flight Price Prediction App</h1>
    <h3>Machine Learning based Fare Prediction</h3>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------
# Prediction Button
# -------------------------------------------------
if st.sidebar.button("🚀 Predict Fare"):
    with st.spinner("Predicting flight price..."):
        time.sleep(1)
        input_df = prepare_input()
        prediction = model.predict(input_df)[0]

    st.success(f"💰 Predicted Flight Price: ₹ {prediction:.2f}")

    st.markdown(
        f"""
        <div style="background-color: rgba(0,119,182,0.7);
                    padding: 15px;
                    border-radius: 10px;">
        <h3>🛫 Prediction Summary</h3>
        <p><b>Airline:</b> {airline}</p>
        <p><b>Route:</b> {source} → {destination}</p>
        <p><b>Journey Date:</b> {journey_date}</p>
        <p><b>Total Stops:</b> {total_stops}</p>
        <p><b>Predicted Price:</b> ₹ {prediction:.2f}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown(
    """
    ---
    💡 *Prices vary based on seasonality, demand, airline, and route.*  
    Built using **Python, Scikit-Learn & Streamlit**
    """
)
