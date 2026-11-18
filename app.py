import streamlit as st
import pandas as pd
import joblib
import json
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_crop_model.joblib"
METRICS_PATH = BASE_DIR / "models" / "model_comparison.csv"

# Page configuration
st.set_page_config(
    page_title="Weather-Based Crop Prediction",
    page_icon="🌱",
    layout="centered"
)

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# Load model evaluation results
metrics = pd.read_csv(METRICS_PATH)

# Website heading
st.title("🌱 Weather-Based Crop Prediction System")

st.write(
    "This machine learning application predicts a suitable "
    "crop based on weather and soil parameters."
)

st.divider()

st.subheader("Enter Weather and Soil Information")

st.info(
    "Enter the seven values below to receive a crop prediction."
)

# Input form
with st.form("crop_prediction_form"):

    col1, col2 = st.columns(2)

    with col1:
        nitrogen = st.number_input(
            "Nitrogen (N)",
            min_value=0.0,
            max_value=200.0,
            value=50.0
        )

        phosphorus = st.number_input(
            "Phosphorus (P)",
            min_value=0.0,
            max_value=200.0,
            value=50.0
        )

        potassium = st.number_input(
            "Potassium (K)",
            min_value=0.0,
            max_value=250.0,
            value=50.0
        )

        temperature = st.number_input(
            "Temperature (°C)",
            min_value=0.0,
            max_value=60.0,
            value=25.0
        )

    with col2:
        humidity = st.number_input(
            "Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=70.0
        )

        ph = st.number_input(
            "Soil pH",
            min_value=0.0,
            max_value=14.0,
            value=6.5
        )

        rainfall = st.number_input(
            "Rainfall (mm)",
            min_value=0.0,
            max_value=500.0,
            value=100.0
        )

    submitted = st.form_submit_button(
        "Predict Suitable Crop",
        use_container_width=True
    )

# Prediction
if submitted:

    input_data = pd.DataFrame([{
        "N": nitrogen,
        "P": phosphorus,
        "K": potassium,
        "temperature": temperature,
        "humidity": humidity,
        "ph": ph,
        "rainfall": rainfall
    }])

    prediction = model.predict(input_data)[0]

    st.success("Prediction completed successfully!")

    st.subheader("🌾 Recommended Crop")
    st.markdown(f"## {prediction.capitalize()}")

    st.caption(
        "This is a machine-learning prediction, not a guarantee "
        "of crop suitability or agricultural success."
    )

    # Show model confidence scores, if supported
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_data)[0]

        probability_data = pd.DataFrame({
            "Crop": model.classes_,
            "Prediction Score (%)": probabilities * 100
        })

        probability_data = probability_data.sort_values(
            by="Prediction Score (%)",
            ascending=False
        )

        st.subheader("Top 5 Crop Prediction Scores")

        st.dataframe(
            probability_data.head(5),
            hide_index=True,
            use_container_width=True
        )

# Model performance
st.divider()
st.subheader("Machine Learning Model Performance")

st.dataframe(
    metrics,
    hide_index=True,
    use_container_width=True
)

st.caption(
    "Performance metrics were calculated using the held-out "
    "test dataset."
)

st.info(
    "Note: This dataset includes nitrogen, phosphorus, potassium "
    "and soil pH along with weather measurements. Actual crop "
    "suitability also depends on location, soil conditions, "
    "irrigation and farming practices."
)