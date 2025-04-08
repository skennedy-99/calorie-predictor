import streamlit as st
import requests

API_URL = "http://backend-container:8000/"


st.set_page_config(page_title="Calorie Burn Predictor", layout="centered")
st.title("🔥 Calorie Burn Predictor")
st.write("Enter your details below to predict calories burnt during exercise.")


col1, col2 = st.columns(2)

with col1:
    gender_option = st.radio(
        "Gender",
        ("Male", "Female"),
        horizontal=True 
    )

    gender = 0 if gender_option == "Male" else 1

    age = st.number_input(
        "Age (years)",
        min_value=1,
        max_value=120,
        value=25,
        step=1
    )
    height = st.number_input(
        "Height (cm)",
        min_value=50.0,
        max_value=250.0,
        value=170.0,
        step=0.5,
        format="%.1f"
    )
    weight = st.numer_input(
        "Weight (kgs)",
        min_value=36,
        max_value=132,
        value=70,
        step=1,
        format="%.1f"
    )

with col2:
    heart_rate = st.number_input(
        "Average Heart Rate (bpm)",
        min_value=40.0,
        max_value=220.0,
        value=90.0,
        step=1.0,
        format="%.1f"
    )
    body_temp = st.number_input(
        "Body Temperature (°C)",
        min_value=35.0,
        max_value=45.0,
        value=39.0,
        step=0.1,
        format="%.1f"
    )
    Duration = st.number_input(
        "Workout Duration (mins)", 
        min_value = 1,
        max_value = 30,
        value = 15,
        step=1,
        format="%.1f"
    )

BMI = weight / ((height/100) **2)

st.divider() 

if st.button("Predict Calories Burnt", type="primary"):
    payload = {
        "Heart_Rate": float(heart_rate),
        "Body_Temp": float(body_temp),
        "Duration": float(Duration),
        "BMI": float(BMI),
        "Gender": gender,
        "Age": float(age)
    }

    with st.spinner('Sending data to prediction API...'):
        try:
            response = requests.post(API_URL, json=payload, timeout=30)

            if response.status_code == 200:
                result = response.json()
                prediction = result.get("prediction")

                if prediction is not None:
                    st.metric(label="Predicted Calories Burnt", value=f"{prediction:.2f} cal")
                    st.success("Prediction successful!")
                else:
                    st.error("API response received, but prediction value is missing.")
                    st.json(result)

            else:
                st.error(f"API Error: Status Code {response.status_code}")

        except requests.exceptions.RequestException as e:
            st.error(f"Network Error: Could not connect to the API.")
            st.error(f"Details: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
