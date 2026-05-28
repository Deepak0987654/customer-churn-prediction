import streamlit as st
import pandas as pd
import joblib

# ---------- LOAD ----------
model = joblib.load("outputs/customer_churn_model.pkl")
scaler = joblib.load("outputs/scaler.pkl")
features = joblib.load("outputs/features.pkl")

# 🔥 Load dataset (IMPORTANT)
data = pd.read_csv("data/Telco-Customer-Churn.csv")

st.set_page_config(layout="wide")

# ---------- SIDEBAR ----------
st.sidebar.title("🎛️ Customer Input")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior = st.sidebar.selectbox("Senior", ["Yes", "No"])
tenure = st.sidebar.slider("Tenure", 0, 72, 12)
monthly = st.sidebar.slider("Monthly Charges", 0.0, 200.0, 50.0)

contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

phone = st.sidebar.selectbox("Phone", ["Yes", "No"])
if phone == "Yes":
    multiple = st.sidebar.selectbox("Multiple Lines", ["Yes", "No"])
else:
    multiple = "No phone service"
    st.sidebar.selectbox("Multiple Lines", ["No phone service"], disabled=True)

internet = st.sidebar.selectbox("Internet", ["DSL", "Fiber optic", "No"])
if internet == "No":
    online_sec = "No internet service"
    st.sidebar.selectbox("Online Security", ["No internet service"], disabled=True)
    tech_sup = "No internet service"
    st.sidebar.selectbox("Tech Support", ["No internet service"], disabled=True)
else:
    online_sec = st.sidebar.selectbox("Online Security", ["Yes", "No"])
    tech_sup = st.sidebar.selectbox("Tech Support", ["Yes", "No"])

# ---------- PREP ----------
def prepare_input():
    df = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "tenure": tenure,
        "MonthlyCharges": monthly,
        "Contract": contract,
        "InternetService": internet,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "OnlineSecurity": online_sec,
        "TechSupport": tech_sup
    }])

    df = pd.get_dummies(df)

    for col in features:
        if col not in df.columns:
            df[col] = 0

    return df[features]

# ---------- KPI (STATIC) ----------
st.title("📊 Customer 360 Churn Dashboard")

k1, k2, k3 = st.columns(3)

k1.metric("Total Customers", len(data))
data['Churn'] = data['Churn'].replace({'Yes': 1, 'No': 0}).astype(float)
k2.metric("Churn Rate", f"{data['Churn'].mean()*100:.1f}%")
k3.metric("Avg Charges", f"${data['MonthlyCharges'].mean():.0f}")

# ---------- CHARTS ----------
c1, c2, c3 = st.columns(3)

with c1:
    st.subheader("Churn Distribution")
    st.bar_chart(data["Churn"].value_counts())

with c2:
    st.subheader("Contract vs Churn")
    contract_churn = pd.crosstab(data["Contract"], data["Churn"])
    st.bar_chart(contract_churn)

with c3:
    st.subheader("Tenure vs Churn")
    st.line_chart(data.groupby("tenure")["Churn"].mean())

# ---------- MODEL ----------
input_df = prepare_input()
scaled = scaler.transform(input_df)
prob = model.predict_proba(scaled)[0][1]

# ---------- PREDICTION ----------
st.markdown("## 🤖 Prediction")

p1, p2, p3 = st.columns(3)

p1.metric("Churn Probability", f"{prob*100:.1f}%")

if prob > 0.7:
    risk = "🔴 High"
elif prob > 0.4:
    risk = "🟡 Medium"
else:
    risk = "🟢 Low"

p2.metric("Risk Level", risk)

clv = monthly * tenure
p3.metric("Customer Value", f"${clv:.0f}")

# ---------- PROGRESS ----------
st.progress(float(prob))

# ---------- INSIGHTS ----------
if prob > 0.7:
    st.error("⚠️ High risk customer. Offer retention plan.")
elif prob > 0.4:
    st.warning("⚠️ Moderate risk. Monitor closely.")
else:
    st.success("✅ Customer likely to stay.")