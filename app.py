import streamlit as st
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Performance Prediction System")
st.write("Predict student performance using Artificial Intelligence and Machine Learning.")
st.divider()

# Load student dataset
data_path = Path(__file__).parent / "student_data.csv"
data = pd.read_csv(data_path)

# Train the Machine Learning model
features = [
    "study_hours",
    "attendance",
    "previous_marks",
    "assignments",
    "participation_score"
]

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

import numpy as np

np.random.seed(42)
n = 300
low = pd.DataFrame({
    "study_hours": np.random.uniform(0, 3, n),
    "attendance": np.random.uniform(0, 50, n),
    "previous_marks": np.random.uniform(0, 50, n),
    "assignments": np.random.uniform(0, 50, n),
    "participation_score": np.random.choice([0, 1], n),
})
low["performance_score"] = np.clip(
    low["study_hours"] * 2
    + low["attendance"] * 0.12
    + low["previous_marks"] * 0.15
    + low["assignments"] * 0.1
    + np.random.normal(0, 2, n),
    0, 100,
)

data = pd.concat([data, low], ignore_index=True)
model.fit(data[features], data["performance_score"])

st.header("📚 Enter Student Details")

col1, col2 = st.columns(2)

with col1:
    study_hours = st.slider(
        "Daily Study Hours", 0.0, 12.0, 4.0, 0.5
    )
    attendance = st.slider(
        "Attendance (%)", 0, 100, 75
    )
    previous_marks = st.slider(
        "Previous Exam Marks (%)", 0, 100, 65
    )

with col2:
    assignments = st.slider(
        "Assignments Completed (%)", 0, 100, 80
    )
    participation = st.selectbox(
        "Class Participation", ["Low", "Medium", "High"]
    )

participation_score = {
    "Low": 40,
    "Medium": 70,
    "High": 90
}[participation]

if st.button(
    "🔮 Predict Student Performance",
    use_container_width=True
):
    student = pd.DataFrame([{
        "study_hours": study_hours,
        "attendance": attendance,
        "previous_marks": previous_marks,
        "assignments": assignments,
        "participation_score": participation_score
    }])

    prediction = model.predict(student[features])[0]
    prediction = max(0, min(100, prediction))
    if (study_hours == 0 and attendance == 0 and previous_marks == 0
        and assignments == 0 ):
        prediction = 0.0

    st.subheader("📊 Prediction Result")
    st.metric(
        "Predicted Performance Score",
        f"{prediction:.1f}%"
    )

    if prediction >= 90:
        st.success("Outstanding predicted performance! Keep it up.")

    elif prediction >= 75:
        st.success("Excellent predicted performance. Continue your consistent efforts.")

    elif prediction >= 60:
        st.info("Good progress! Focus on improving weaker areas.")

    elif prediction >= 45:
        st.info("Moderate predicted performance. More consistent preparation may help.")

    elif prediction >= 30:
        st.warning("Below-average predicted performance. Focus on regular study and assignments.")

    elif prediction >= 15:
        st.warning("Low predicted performance. Strengthen your preparation and seek academic support.")

    elif prediction >= 5:
        st.error("Very low predicted performance. Significant improvement in study habits may help.")

    else:
        st.error("Extremely low predicted performance. Additional academic support is recommended.")

st.divider()
st.caption(
    "Third Year IT | AI & ML Mini Project | "
    "Predictions are estimates based on sample training data."
)