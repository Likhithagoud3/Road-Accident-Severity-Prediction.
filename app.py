import streamlit as st
import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pydeck as pdk

# -------------------------------
# FILE PATHS
# -------------------------------
CSV_PATH = "road_accidents.csv"
MODEL_PATH = "accident_severity_model.pkl"

FEATURES = [
    "Time", "Day_of_week", "Age_band_of_driver", "Sex_of_driver",
    "Educational_level", "Type_of_vehicle", "Area_accident_occured",
    "Road_allignment", "Weather_conditions", "Light_conditions",
    "Type_of_collision", "Number_of_vehicles_involved", "Number_of_casualties"
]

# -------------------------------
# MODEL TRAINING FUNCTION
# -------------------------------
def train_model():
    df = pd.read_csv(CSV_PATH)
    df = df[FEATURES + ["Accident_severity"]]

    le = LabelEncoder()
    for col in df.columns:
        df[col] = le.fit_transform(df[col])

    X = df[FEATURES]
    y = df["Accident_severity"]

    model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    model.fit(X, y)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "features": FEATURES}, f)

    return model

# -------------------------------
# LOAD OR TRAIN MODEL
# -------------------------------
if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, dict):
            model = data["model"]
            FEATURES = data["features"]
        else:
            model = data
            st.warning("⚠ Loaded model without metadata.")
    except:
        st.error("❌ Model corrupted. Training new one...")
        model = train_model()
else:
    st.warning("Model not found. Training new one...")
    model = train_model()

# -------------------------------
# STREAMLIT UI
# -------------------------------
st.title("🚧 Road Accident Severity Prediction App")
st.markdown("Predict the severity of road accidents and get precautionary advice.")

st.sidebar.title("🚦 Accident Details Input")

# ----------- USER INPUTS ----------
time = st.sidebar.selectbox("⏰ Time of Day", ["Morning", "Afternoon", "Evening", "Night"])
day = st.sidebar.selectbox("📅 Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
age_band = st.sidebar.selectbox("🧑 Age Band", ["Under 18", "18-30", "31-50", "Over 50"])
sex = st.sidebar.selectbox("🚻 Driver Sex", ["Male", "Female"])
education = st.sidebar.selectbox("🎓 Education Level", ["None", "Primary", "Secondary", "University"])
vehicle = st.sidebar.selectbox("🚗 Vehicle Type", ["Car", "Motorcycle", "Bus", "Truck", "Bicycle"])
area = st.sidebar.selectbox("🌍 Accident Area", ["Urban", "Rural"])
alignment = st.sidebar.selectbox("🛣 Road Alignment", ["Straight", "Curved"])
weather = st.sidebar.selectbox("🌦 Weather Conditions", ["Clear", "Cloudy", "Rainy", "Foggy"])
light = st.sidebar.selectbox("💡 Light Conditions", ["Daylight", "Night - Lighted", "Night - Dark"])
collision = st.sidebar.selectbox("💥 Collision Type", ["Front", "Rear", "Side", "Multiple"])
vehicles = st.sidebar.slider("🚘 Vehicles Involved", 1, 5, 1)
casualties = st.sidebar.slider("🩸 Number of Casualties", 0, 10, 1)

# ----------- FORCE FATAL CHECKBOX (for testing) ----------
force_fatal = st.sidebar.checkbox("⚠ Force Fatal Prediction (Test Only)")

# ----------- MAPPING INPUTS ----------
time_map = {"Morning": 0, "Afternoon": 1, "Evening": 2, "Night": 3}
day_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
           "Friday": 4, "Saturday": 5, "Sunday": 6}
age_map = {"Under 18": 0, "18-30": 1, "31-50": 2, "Over 50": 3}
sex_map = {"Male": 0, "Female": 1}
edu_map = {"None": 0, "Primary": 1, "Secondary": 2, "University": 3}
vehicle_map = {"Car": 0, "Motorcycle": 1, "Bus": 2, "Truck": 3, "Bicycle": 4}
area_map = {"Urban": 0, "Rural": 1}
align_map = {"Straight": 0, "Curved": 1}
weather_map = {"Clear": 0, "Cloudy": 1, "Rainy": 2, "Foggy": 3}
light_map = {"Daylight": 0, "Night - Lighted": 1, "Night - Dark": 2}
collision_map = {"Front": 0, "Rear": 1, "Side": 2, "Multiple": 3}

input_list = [
    time_map[time], day_map[day], age_map[age_band], sex_map[sex], edu_map[education],
    vehicle_map[vehicle], area_map[area], align_map[alignment], weather_map[weather],
    light_map[light], collision_map[collision], vehicles, casualties
]

input_df = pd.DataFrame([input_list], columns=FEATURES)

# -------------------------------
# PREDICTION
# -------------------------------
if st.button("Predict Severity"):
    if force_fatal:
        prediction = 0
        confidence = 99.9
        st.success("⚠ Fatal prediction forced for testing!")
    else:
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]
        confidence = max(prediction_proba) * 100

    # Severity info
    severity_data = {
        0: {"label": "🔴 Fatal Injury Accident", "color": "#FF8B94",
            "precautions": ["🚨 Call emergency services immediately",
                            "🛑 Avoid risky maneuvers or shortcuts",
                            "⚠ Ensure proper traffic management if on-site",
                            "✅ Follow all traffic rules strictly"]},
        1: {"label": "🟠 Serious Injury Accident", "color": "#FFD3B6",
            "precautions": ["⚠ Drive cautiously and maintain safe speed",
                            "📵 Avoid distractions (phone, eating, etc.)",
                            "🪖 Wear safety equipment (seatbelt, helmet)",
                            "✅ Follow traffic signals and regulations carefully"]},
        2: {"label": "🟢 Slight Injury Accident", "color": "#A8E6CF",
            "precautions": ["✅ Maintain normal driving precautions",
                            "🚦 Follow traffic rules",
                            "👀 Be attentive to surroundings"]}
    }

    data = severity_data[prediction]

    # Display severity
    st.markdown(
        f"<div style='background-color: {data['color']}; padding: 20px; border-radius: 10px;'>"
        f"<h3>{data['label']}</h3>"
        f"<p><b>Confidence:</b> {confidence:.2f}%</p>"
        f"<p><b>Precautions & Advice:</b></p>"
        f"{''.join([f'<li>{p}</li>' for p in data['precautions']])}"
        f"</div>",
        unsafe_allow_html=True
    )

    st.progress(int(confidence))

    # Extra safety tips
    with st.expander("More Safety Tips"):
        st.write("- Avoid phone usage while driving")
        st.write("- Maintain safe distance from other vehicles")
        st.write("- Check weather and road conditions before travel")
        st.write("- Keep first aid kit and emergency contacts handy")

# -------------------------------


