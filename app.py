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
st.write("Predict student performance using Machine Learning.")
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
    "sleep_hours",
    "participation_score"
]

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

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
    sleep_hours = st.slider(
        "Daily Sleep Hours", 0.0, 12.0, 7.0, 0.5
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
        "sleep_hours": sleep_hours,
        "participation_score": participation_score
    }])

    prediction = model.predict(student[features])[0]
    prediction = max(0, min(100, prediction))

    st.subheader("📊 Prediction Result")
    st.metric(
        "Predicted Performance Score",
        f"{prediction:.1f}%"
    )

    if prediction >= 75:
        st.success("Excellent performance!")
    elif prediction >= 50:
        st.info("Good progress! Keep improving.")
    else:
        st.warning("More preparation may help.")

st.divider()
st.caption(
    "Third Year IT | AI & ML Mini Project | "
    "Predictions are estimates based on sample training data."
)