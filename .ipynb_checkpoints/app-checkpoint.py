# app.py - Flight Price Predictor (Final Version with Results & Graphs)
import streamlit as st
import joblib
import pandas as pd
import numpy as np
from PIL import Image
import plotly.express as px
import os

# 🎨 Page Config
st.set_page_config(
    layout="wide",
    page_title="✈️ Flight Price Predictor",
    page_icon="✈️"
)

# 🔹 Project Folder
project_folder = os.path.dirname(__file__)  # folder where app.py is

# 🔹 Load model from multiple possible locations
possible_paths = [
    os.path.join(project_folder, "flight_model.pkl"),
    os.path.join(project_folder, "flight_price_model.pkl"),
    r"C:\Users\khkar\Documents\flight_price_model.pkl",
    r"C:\Users\khkar\Flight fare prediction\flight_price_model.pkl"
]

model = None
for path in possible_paths:
    if os.path.exists(path):
        try:
            model = joblib.load(path)
            st.sidebar.success(f"✅ Model loaded from: {path}")
            break
        except Exception as e:
            st.sidebar.error(f"⚠️ Error loading model from {path}: {e}")

if model is None:
    st.sidebar.error("❌ No model file found! Please place your `.pkl` file correctly.")

# 🎨 Add Header Image
img_path = os.path.join(project_folder, "pic1.png")
if os.path.exists(img_path):
    st.image(img_path, use_container_width=True)
else:
    st.warning("⚠️ Header image not found (pic1.png). Skipping banner...")

# Title & Description
st.title("✨ AI-Powered Flight Price Prediction")
st.markdown("Predict your flight ticket price smartly with ML 💡")

# Sidebar ✈️
st.sidebar.header("🛫 Enter Flight Details")
st.sidebar.markdown("---")

# Inputs
airline = st.sidebar.selectbox("✈️ Airline", 
                               ["IndiGo","Air India","SpiceJet","Vistara","GoAir","Jet Airways"])

source = st.sidebar.selectbox("🛬 Source", ["Delhi","Kolkata","Mumbai","Chennai","Banglore"])
destination = st.sidebar.selectbox("🏙️ Destination", ["Cochin","Delhi","Hyderabad","Kolkata","New Delhi"])
route = st.sidebar.text_input("🛤️ Route", value="DEL → BOM → COK")
additional_info = st.sidebar.radio("ℹ️ Additional Info", 
                                   ["No info","In-flight meal not included","No check-in baggage included"])

dep_date = st.sidebar.date_input("📅 Departure Date")
dep_time = st.sidebar.time_input("🕒 Departure Time")

duration_hours = st.sidebar.number_input("⏳ Duration (hours)", min_value=0, max_value=48, value=2)
duration_mins = st.sidebar.slider("⏱️ Duration (minutes)", 0, 59, 30)

total_stops = st.sidebar.slider("🛑 Total Stops", 0, 4, 1)

# 🚀 Prepare input data
input_dict = {
    "Duration_mins": duration_hours*60 + duration_mins,
    "Total_Stops_Num": total_stops,
    "Journey_Day": dep_date.day,
    "Journey_Month": dep_date.month,
    "Dep_Hour": dep_time.hour,
    "Dep_Min": dep_time.minute,
    "Airline": airline,
    "Source": source,
    "Destination": destination,
    "Route": route,
    "Additional_Info": additional_info
}

input_df = pd.DataFrame([input_dict])

# Store prediction history
if "history" not in st.session_state:
    st.session_state["history"] = pd.DataFrame()

# Layout with columns
col1, col2 = st.columns([2,1])

with col1:
    st.subheader("🔍 Input Preview")
    st.dataframe(input_df)

with col2:
    if st.button("🚀 Predict Price"):
        if model:
            try:
                pred = model.predict(input_df)[0]
                st.success(f"💰 Predicted Flight Price: ₹{pred:,.2f}")
                st.balloons()

                # Save to history
                result_row = input_df.copy()
                result_row["Predicted_Price"] = pred
                st.session_state["history"] = pd.concat([st.session_state["history"], result_row], ignore_index=True)

            except Exception as e:
                st.error(f"⚠️ Prediction failed: {e}")
        else:
            st.error("❌ Model not loaded. Please check your `.pkl` file path.")

# 📊 Results Section
if not st.session_state["history"].empty:
    st.markdown("---")
    st.subheader("📊 Prediction Results History")
    st.dataframe(st.session_state["history"])

    # Graph 1: Duration vs Predicted Price
    fig1 = px.scatter(st.session_state["history"],
                     x="Duration_mins", y="Predicted_Price",
                     size="Total_Stops_Num", color="Airline",
                     title="Flight Duration vs Predicted Price",
                     hover_data=["Source", "Destination"])
    st.plotly_chart(fig1, use_container_width=True)

    # Graph 2: Journey Month vs Avg Price
    fig2 = px.bar(st.session_state["history"].groupby("Journey_Month")["Predicted_Price"].mean().reset_index(),
                 x="Journey_Month", y="Predicted_Price",
                 title="Average Price per Month", text="Predicted_Price")
    st.plotly_chart(fig2, use_container_width=True)

# 📊 Feature Importance Section
st.markdown("---")
st.subheader("📊 Feature Importance (ML Insights)")

fi_path = os.path.join(project_folder, "feature_importance.xlsx")
if os.path.exists(fi_path):
    fi = pd.read_excel(fi_path)
    fig = px.bar(fi.head(15).sort_values('Feature Importance Score', ascending=True),
                 x='Feature Importance Score', y='Variable',
                 orientation='h', text='Feature Importance Score',
                 color='Feature Importance Score', color_continuous_scale="blues")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("ℹ️ Feature importance data not available")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit & Machine Learning")
