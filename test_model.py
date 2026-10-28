import pandas as pd
from sklearn.ensemble import RandomForestRegressor

data = pd.read_csv("student_data.csv")

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

student = pd.DataFrame([{
"study_hours": 0,
"attendance": 0,
"previous_marks": 0,
"assignments": 0,
"sleep_hours": 0,
"participation_score": 0
}])

prediction = model.predict(student[features])[0]

print("Prediction for all-zero inputs:", round(prediction, 2))
print("Lowest training score:", data["performance_score"].min())
print("Highest training score:", data["performance_score"].max())