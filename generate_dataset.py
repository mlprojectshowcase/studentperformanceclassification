"""
generate_dataset.py
-------------------
Creates a SYNTHETIC (artificial) student dataset and saves it to data/student_data.csv.

Why synthetic?
  Real student records are private. Here we invent 1000 students whose final result
  depends on realistic factors (study hours, attendance, previous marks ...) plus some
  random noise, so the machine-learning models have real patterns to learn.

You can replace data/student_data.csv with a real dataset (for example the UCI
"Student Performance" dataset) as long as the column names stay the same.
Run:  python generate_dataset.py
"""
import numpy as np
import pandas as pd


def make_dataframe():
    """Builds the synthetic dataset and returns it as a DataFrame."""
    rng = np.random.default_rng(42)   # fixed seed -> same data every time
    N = 1000

    gender = rng.choice(["Male", "Female"], N)
    study_hours = np.clip(rng.normal(3.5, 1.5, N), 0, 10).round(1)          # hours per day
    attendance = np.clip(rng.normal(80, 12, N), 35, 100).round(1)           # percent
    previous_marks = np.clip(rng.normal(65, 15, N), 20, 100).round(1)       # last exam, out of 100
    assignments = np.clip(rng.normal(75, 18, N), 10, 100).round(1)          # percent submitted
    sleep_hours = np.clip(rng.normal(7, 1.2, N), 3.5, 10).round(1)
    parent_edu = rng.choice(["High School", "Graduate", "Post Graduate"], N, p=[0.45, 0.40, 0.15])
    internet = rng.choice(["Yes", "No"], N, p=[0.8, 0.2])
    extra = rng.choice(["Yes", "No"], N, p=[0.5, 0.5])                      # extracurricular activities

    # Hidden "true" score: a weighted mix of the factors + random noise
    edu_bonus = pd.Series(parent_edu).map({"High School": 0, "Graduate": 2, "Post Graduate": 4}).values
    score = (
        0.35 * previous_marks
        + 0.22 * attendance
        + 3.0 * study_hours
        + 0.12 * assignments
        + 1.0 * (sleep_hours - 7).clip(-3, 1)
        + edu_bonus
        + 2.0 * (internet == "Yes")
        + rng.normal(0, 5, N)
    )

    # Convert score into 3 classes using percentiles so classes are reasonably balanced
    low_cut, high_cut = np.percentile(score, [30, 70])
    performance = np.where(score < low_cut, "Low", np.where(score < high_cut, "Medium", "High"))

    df = pd.DataFrame({
        "gender": gender,
        "study_hours_per_day": study_hours,
        "attendance_percent": attendance,
        "previous_marks": previous_marks,
        "assignments_completed_percent": assignments,
        "sleep_hours": sleep_hours,
        "parental_education": parent_edu,
        "internet_access": internet,
        "extracurricular": extra,
        "performance": performance,
    })
    return df


if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    df = make_dataframe()
    df.to_csv("data/student_data.csv", index=False)
    print("Saved data/student_data.csv ->", df.shape)
    print(df["performance"].value_counts())
