"""
ml_pipeline.py
--------------
All the machine-learning logic lives here, so that BOTH train_model.py (command line)
and app.py (Streamlit website) use exactly the same code.

Steps:
  1. load_data()          -> read the CSV file
  2. build_preprocessor() -> convert text columns to numbers + scale numbers
  3. get_models()         -> the 3 algorithms we compare
  4. train_and_evaluate() -> train, test, and return accuracy / reports
"""
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

DATA_PATH = "data/student_data.csv"
TARGET = "performance"
LABELS = ["Low", "Medium", "High"]          # order used in reports / charts

NUMERIC = ["study_hours_per_day", "attendance_percent", "previous_marks",
           "assignments_completed_percent", "sleep_hours"]
CATEGORICAL = ["gender", "parental_education", "internet_access", "extracurricular"]


def load_data(path=DATA_PATH):
    return pd.read_csv(path)


def build_preprocessor():
    """Numbers are scaled (same range); text columns are one-hot encoded (0/1 columns)."""
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])


def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    }


def train_and_evaluate(df):
    """Train all models on 80% of data, test on the remaining 20%."""
    X, y = df[NUMERIC + CATEGORICAL], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    results, fitted = {}, {}
    for name, model in get_models().items():
        pipe = Pipeline([("prep", build_preprocessor()), ("model", model)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, pred, average="weighted", zero_division=0)
        cv = cross_val_score(pipe, X_train, y_train, cv=5).mean()   # 5-fold cross validation
        results[name] = {
            "accuracy": accuracy_score(y_test, pred),
            "precision": prec, "recall": rec, "f1": f1,
            "cv_accuracy": cv,
            "confusion": confusion_matrix(y_test, pred, labels=LABELS),
        }
        fitted[name] = pipe

    best = max(results, key=lambda n: results[n]["accuracy"])
    return fitted, results, best, (X_train, X_test, y_train, y_test)


def feature_importance(pipe):
    """Which inputs matter most? (works for Random Forest / Decision Tree)"""
    names = pipe.named_steps["prep"].get_feature_names_out()
    names = [n.split("__")[1] for n in names]
    imp = pipe.named_steps["model"].feature_importances_
    return pd.Series(imp, index=names).sort_values(ascending=False)
