import streamlit as st
import pandas as pd
import pickle

# =========================
# LOAD MODEL + SCALER + COLUMNS
# =========================
with open('telecom_churn_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('columns.pkl', 'rb') as file:
    model_columns = pickle.load(file)

# =========================
# UI TITLE
# =========================
st.title("📞 Telecom Customer Churn Prediction App")

st.markdown("Enter customer details below to predict churn risk.")

# =========================
# USER INPUT (CLEAN VERSION)
# =========================

gender = st.selectbox("Gender", ["Female", "Male"])
senior_citizen = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partner", ["No", "Yes"])
dependents = st.selectbox("Dependents", ["No", "Yes"])

tenure = st.slider("Tenure (Months)", 0, 72, 12)

contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

phone_service = st.selectbox("Phone Service", ["No", "Yes"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
total_charges = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)

paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])

payment_method = st.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)

# =========================
# FEATURE ENGINEERING
# =========================
input_data = pd.DataFrame([{
    "gender": 1 if gender == "Male" else 0,
    "SeniorCitizen": senior_citizen,
    "Partner": 1 if partner == "Yes" else 0,
    "Dependents": 1 if dependents == "Yes" else 0,
    "tenure": tenure,
    "PhoneService": 1 if phone_service == "Yes" else 0,
    "InternetService_DSL": 1 if internet_service == "DSL" else 0,
    "InternetService_Fiber optic": 1 if internet_service == "Fiber optic" else 0,
    "InternetService_No": 1 if internet_service == "No" else 0,
    "MonthlyCharges": monthly_charges,
    "Total_Charges": total_charges,
    "PaperlessBilling": 1 if paperless_billing == "Yes" else 0,
    "Contract_Month-to-month": 1 if contract == "Month-to-month" else 0,
    "Contract_One year": 1 if contract == "One year" else 0,
    "Contract_Two year": 1 if contract == "Two year" else 0,
    "PaymentMethod_Electronic check": 1 if payment_method == "Electronic check" else 0,
    "PaymentMethod_Mailed check": 1 if payment_method == "Mailed check" else 0,
    "PaymentMethod_Bank transfer (automatic)": 1 if payment_method == "Bank transfer (automatic)" else 0,
    "PaymentMethod_Credit card (automatic)": 1 if payment_method == "Credit card (automatic)" else 0
}])

# =========================
# ALIGN COLUMNS (VERY IMPORTANT)
# =========================
input_data = input_data.reindex(columns=model_columns, fill_value=0)

# =========================
# SCALING
# =========================
input_scaled = scaler.transform(input_data)

# =========================
# PREDICTION
# =========================
if st.button("🔍 Predict Churn"):

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    # =========================
    # OUTPUT
    # =========================
    if prediction == 1:
        st.error("⚠️ Customer is likely to CHURN")
    else:
        st.success("✅ Customer is likely to STAY")

    st.write(f"### 🔥 Churn Probability: {probability:.2f}")