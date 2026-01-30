# ✈️ Flight Fare Prediction Project

This project predicts flight ticket prices using Machine Learning and also forecasts future price trends using time-series analysis.

---

## 🔍 Project Overview

The system includes:
- A **Machine Learning regression model** for predicting flight prices
- **Feature importance analysis** using tree-based models
- **Time-series forecasting** using Facebook Prophet
- An interactive **Streamlit web application**

---

## 🧠 Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- Prophet (Meta)
- Streamlit
- Matplotlib / Plotly
- Git & GitHub

---

## 📊 Machine Learning Model

- Input features:
  - Airline
  - Source & Destination
  - Total Stops
  - Journey Day & Month
  - Departure & Arrival Time
  - Flight Duration
- Output:
  - Predicted flight ticket price

---

## 📈 Time Series Forecasting (Prophet)

- Used journey date and average ticket price
- Forecasts future flight price trends
- Visualized using line charts

---

## 🌐 Streamlit Application

Features:
- User-friendly UI for entering flight details
- Real-time flight price prediction
- Visualization of Prophet forecast trends

Run the app using:
```bash
python -m streamlit run app.py
