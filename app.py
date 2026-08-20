import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

st.title("Credit Card Approval Predictor")

@st.cache_resource
def train_model():
    df = pd.read_csv("Credit_Card_Approval_10000_70_30.csv")
    df = df.drop(columns=["application_id", "application_date"])
    X = df.drop(columns=["approval_status"])
    y = df["approval_status"]

    numeric_features = ["age", "annual_income", "credit_score", "loan_amount",
                         "years_employed", "debt_to_income_ratio", "existing_loans"]
    categorical_features = ["employment_type", "marital_status", "education_level",
                             "residence_type", "city_tier", "card_type_requested"]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    model.fit(X_train, y_train)

    df_options = {
        "employment_type": df["employment_type"].dropna().unique().tolist(),
        "marital_status": df["marital_status"].dropna().unique().tolist(),
        "education_level": df["education_level"].dropna().unique().tolist(),
        "residence_type": df["residence_type"].dropna().unique().tolist(),
        "city_tier": df["city_tier"].dropna().unique().tolist(),
        "card_type_requested": df["card_type_requested"].dropna().unique().tolist(),
    }
    return model, df_options

model, options = train_model()

st.header("Enter Applicant Details")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    annual_income = st.number_input("Annual Income", min_value=0, value=50000)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
    loan_amount = st.number_input("Loan Amount", min_value=0, value=10000)
    years_employed = st.number_input("Years Employed", min_value=0, value=3)
    debt_to_income_ratio = st.number_input("Debt to Income Ratio", min_value=0.0, max_value=1.0, value=0.3, step=0.01)
    existing_loans = st.number_input("Existing Loans", min_value=0, value=1)

with col2:
    employment_type = st.selectbox("Employment Type", options["employment_type"])
    marital_status = st.selectbox("Marital Status", options["marital_status"])
    education_level = st.selectbox("Education Level", options["education_level"])
    residence_type = st.selectbox("Residence Type", options["residence_type"])
    city_tier = st.selectbox("City Tier", options["city_tier"])
    card_type_requested = st.selectbox("Card Type Requested", options["card_type_requested"])

if st.button("Predict Approval"):
    input_df = pd.DataFrame([{
        "age": age, "annual_income": annual_income, "credit_score": credit_score,
        "loan_amount": loan_amount, "years_employed": years_employed,
        "debt_to_income_ratio": debt_to_income_ratio, "existing_loans": existing_loans,
        "employment_type": employment_type, "marital_status": marital_status,
        "education_level": education_level, "residence_type": residence_type,
        "city_tier": city_tier, "card_type_requested": card_type_requested
    }])
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    if prediction == "Approved":
        st.success(f"Prediction: {prediction}")
    else:
        st.error(f"Prediction: {prediction}")

    st.write(f"Confidence: {max(probability):.1%}")