"""
app.py  -  Streamlit web app for "Student Performance Classification"
Run locally:  streamlit run app.py
"""
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import ml_pipeline as ml

st.set_page_config(page_title="Student Performance Classification", page_icon="🎓", layout="wide")


@st.cache_data
def get_data():
    return ml.load_data()


@st.cache_resource
def get_models():
    """Trains once when the app starts (takes a few seconds), then reuses the result."""
    return ml.train_and_evaluate(get_data())


df = get_data()
fitted, results, best, _ = get_models()

st.sidebar.title("🎓 Navigation")
page = st.sidebar.radio("Go to", ["1. Project Overview", "2. Dataset Explorer",
                                  "3. Model Results", "4. Predict a Student"])

# ------------------------------------------------------------------ PAGE 1
if page.startswith("1"):
    st.title("🎓 Student Performance Classification")
    st.write("""
**Goal:** predict whether a student's performance will be **Low, Medium or High**
using study habits and background information.

**Type of problem:** *Supervised multi-class classification* - the computer learns from
past students whose result is already known, then predicts the result for a new student.

**Workflow used in this project**
1. Create / load the dataset
2. Pre-process the data (scale numbers, convert text to numbers)
3. Split into 80 % training and 20 % testing data
4. Train three algorithms: Logistic Regression, Decision Tree, Random Forest
5. Compare them using accuracy, precision, recall, F1-score
6. Use the best model to predict new students
""")
    c1, c2, c3 = st.columns(3)
    c1.metric("Students in dataset", len(df))
    c2.metric("Input features", len(ml.NUMERIC) + len(ml.CATEGORICAL))
    c3.metric("Best model", best)
    st.info("Note: the dataset is synthetic (artificially generated) because real student "
            "records are private. See the README for details.")

# ------------------------------------------------------------------ PAGE 2
elif page.startswith("2"):
    st.title("📊 Dataset Explorer")
    st.subheader("First 10 rows")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Summary statistics")
    st.dataframe(df.describe().round(2), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Class distribution")
        counts = df[ml.TARGET].value_counts().reindex(ml.LABELS)
        st.bar_chart(counts)
    with col2:
        st.subheader("Average of a feature per class")
        feat = st.selectbox("Choose a feature", ml.NUMERIC)
        st.bar_chart(df.groupby(ml.TARGET)[feat].mean().reindex(ml.LABELS))

    st.subheader("Correlation between numeric features")
    corr = df[ml.NUMERIC].corr()
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns, fontsize=8)
    fig.colorbar(im); fig.tight_layout()
    st.pyplot(fig)

# ------------------------------------------------------------------ PAGE 3
elif page.startswith("3"):
    st.title("🏆 Model Results")
    table = pd.DataFrame({
        n: {"Accuracy": r["accuracy"], "Precision": r["precision"], "Recall": r["recall"],
            "F1-score": r["f1"], "5-fold CV accuracy": r["cv_accuracy"]}
        for n, r in results.items()
    }).T.round(3)
    st.dataframe(table, use_container_width=True)
    st.success(f"Best model on the test data: **{best}** "
               f"(accuracy {results[best]['accuracy']:.1%})")

    with st.expander("What do these metrics mean?"):
        st.write("""
- **Accuracy** - out of all test students, how many were classified correctly.
- **Precision** - when the model says a class, how often is it right.
- **Recall** - out of all students truly in a class, how many did the model find.
- **F1-score** - balance of precision and recall.
- **Cross-validation** - training/testing repeated 5 times on different parts of the data to check the result is stable.
""")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"Confusion matrix ({best})")
        cm = results[best]["confusion"]
        fig, ax = plt.subplots(figsize=(4, 3.5))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(3)); ax.set_xticklabels(ml.LABELS)
        ax.set_yticks(range(3)); ax.set_yticklabels(ml.LABELS)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, cm[i, j], ha="center", va="center")
        fig.tight_layout()
        st.pyplot(fig)
        st.caption("Diagonal = correct predictions. Other cells = mistakes.")
    with col2:
        st.subheader("Feature importance (Random Forest)")
        imp = ml.feature_importance(fitted["Random Forest"])
        st.bar_chart(imp)
        st.caption("Longer bar = that input influences the prediction more.")

# ------------------------------------------------------------------ PAGE 4
else:
    st.title("🔮 Predict a Student's Performance")
    st.write(f"Enter the student's details. Prediction is made by the best model: **{best}**.")
    with st.form("predict_form"):
        c1, c2 = st.columns(2)
        with c1:
            study = st.slider("Study hours per day", 0.0, 10.0, 3.5, 0.5)
            attendance = st.slider("Attendance (%)", 30, 100, 80)
            marks = st.slider("Previous exam marks (out of 100)", 20, 100, 65)
            assign = st.slider("Assignments completed (%)", 10, 100, 75)
            sleep = st.slider("Sleep hours per night", 3.5, 10.0, 7.0, 0.5)
        with c2:
            gender = st.selectbox("Gender", ["Male", "Female"])
            edu = st.selectbox("Parental education", ["High School", "Graduate", "Post Graduate"])
            internet = st.selectbox("Internet access at home", ["Yes", "No"])
            extra = st.selectbox("Extracurricular activities", ["Yes", "No"])
        go = st.form_submit_button("Predict")

    if go:
        row = pd.DataFrame([{
            "study_hours_per_day": study, "attendance_percent": attendance,
            "previous_marks": marks, "assignments_completed_percent": assign,
            "sleep_hours": sleep, "gender": gender, "parental_education": edu,
            "internet_access": internet, "extracurricular": extra}])
        model = fitted[best]
        pred = model.predict(row)[0]
        proba = pd.Series(model.predict_proba(row)[0], index=model.classes_).reindex(ml.LABELS)
        {"Low": st.error, "Medium": st.warning, "High": st.success}[pred](
            f"Predicted performance: **{pred}**")
        st.write("Confidence for each class:")
        st.bar_chart(proba)
        if pred == "Low":
            st.write("💡 Tip: improving attendance, assignment completion and daily study time helps most.")
