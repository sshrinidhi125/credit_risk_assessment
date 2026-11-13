
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="AI Credit Risk Prediction",
    page_icon="💳",
    layout="wide"
)

st.title("AI-Based Credit Risk Prediction System")
st.write(
    "An intelligent FinTech application that analyzes "
    "loan applicant information and predicts credit default risk."
)

st.divider()

# ---------------- DATASET LOADING ----------------

st.sidebar.header("Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload Credit Risk CSV",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    try:
        df = pd.read_csv("credit_risk_dataset.csv")
    except FileNotFoundError:
        st.warning(
            "Please upload credit_risk_dataset.csv "
            "or place it in the project folder."
        )
        st.stop()

if "loan_status" not in df.columns:
    st.error("Dataset must contain a loan_status column.")
    st.stop()

# ---------------- DATA PREPROCESSING ----------------

df = df.dropna(subset=["loan_status"])
df = df.drop_duplicates()

X = df.drop(columns=["loan_status"])
y = df["loan_status"].astype(int)

if not set(y.unique()).issubset({0, 1}) or len(y.unique()) < 2:
    st.error("loan_status must contain both 0 and 1.")
    st.stop()

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# ---------------- MODEL TRAINING ----------------

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

# ---------------- DASHBOARD ----------------

st.subheader("Credit Risk Analytics Dashboard")

col1, col2, col3 = st.columns(3)

col1.metric("Total Applicants", len(df))
col2.metric("Model Accuracy", f"{accuracy * 100:.2f}%")
col3.metric(
    "Default Cases",
    int((y == 1).sum())
)

st.divider()

# ---------------- DATA VISUALIZATION ----------------

st.subheader("Dataset Analysis")

chart1, chart2 = st.columns(2)

with chart1:
    st.write("Loan Default Distribution")

    counts = y.value_counts().sort_index()

    fig, ax = plt.subplots()
    ax.bar(
        ["No Default", "Default"],
        [counts.get(0, 0), counts.get(1, 0)]
    )
    ax.set_ylabel("Number of Applicants")
    st.pyplot(fig)

with chart2:
    st.write("Annual Income Distribution")

    if "person_income" in df.columns:
        fig, ax = plt.subplots()
        ax.hist(df["person_income"].dropna(), bins=30)
        ax.set_xlabel("Annual Income")
        ax.set_ylabel("Number of Applicants")
        st.pyplot(fig)

st.divider()

# ---------------- MODEL EVALUATION ----------------

st.subheader("Machine Learning Model Evaluation")

st.write(f"**Algorithm:** Random Forest Classifier")
st.write(f"**Accuracy:** {accuracy * 100:.2f}%")

cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

fig, ax = plt.subplots()
ax.imshow(cm, cmap="Blues")

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["No Default", "Default"])
ax.set_yticklabels(["No Default", "Default"])

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")

for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center")

st.pyplot(fig)

st.text("Classification Report")
st.text(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

st.divider()

# ---------------- FEATURE IMPORTANCE ----------------

st.subheader("Important Credit Risk Factors")

classifier = model.named_steps["classifier"]

feature_names = model.named_steps[
    "preprocessor"
].get_feature_names_out()

importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": classifier.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
).head(10)

fig, ax = plt.subplots()
ax.barh(
    importance["Feature"],
    importance["Importance"]
)
ax.invert_yaxis()
ax.set_xlabel("Importance Score")
ax.set_title("Top 10 Influencing Features")

st.pyplot(fig)

st.divider()

# ---------------- INDIVIDUAL RISK PREDICTION ----------------

st.subheader("Individual Loan Applicant Assessment")

st.write(
    "Enter applicant information to estimate the credit default risk."
)

with st.form("prediction_form"):

    input_data = {}

    cols = st.columns(2)

    for i, column in enumerate(X.columns):

        with cols[i % 2]:

            if column in numeric_features:

                median_value = X[column].median()

                if pd.isna(median_value):
                    median_value = 0

                input_data[column] = st.number_input(
                    column.replace("_", " ").title(),
                    value=float(median_value)
                )

            else:

                options = (
                    X[column]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )

                if len(options) == 0:
                    options = ["Unknown"]

                input_data[column] = st.selectbox(
                    column.replace("_", " ").title(),
                    options
                )

    submitted = st.form_submit_button(
        "Predict Credit Risk"
    )

if submitted:

    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)[0]

    probabilities = model.predict_proba(input_df)[0]

    default_probability = probabilities[
        list(model.named_steps["classifier"].classes_).index(1)
    ]

    st.subheader("Prediction Result")

    st.metric(
        "Estimated Default Probability",
        f"{default_probability * 100:.2f}%"
    )

    if prediction == 1:
        st.error("Higher Credit Default Risk")
    else:
        st.success("Lower Credit Default Risk")

    st.progress(float(default_probability))

    st.caption(
        "This is a machine-learning risk estimate, "
        "not a final lending or loan approval decision."
    )

st.divider()

st.caption(
    "AI-Based Credit Risk Prediction System | "
    "Financial Technologies Foundations"
)
