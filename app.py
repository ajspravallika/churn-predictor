import altair as alt
import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Churn Predictor", page_icon="📉")

# Load the saved model and cutoff (cached so it loads only once)
BASE = Path(__file__).parent

@st.cache_resource
def load_model():
    model = joblib.load(BASE / "churn_model.pkl")
    cutoff = float((BASE / "cutoff.txt").read_text().strip())
    return model, cutoff

model, cutoff = load_model()

st.title("📉 Customer Churn Predictor")
st.write("Enter a customer's details to see how likely they are to cancel.")

# ---- Account details ----
st.subheader("Account")
col1, col2 = st.columns(2)
with col1:
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly = st.number_input("Monthly charges", 18.0, 120.0, 70.0, step=1.0)
with col2:
    payment = st.selectbox("Payment method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"])
    paperless = st.selectbox("Paperless billing", ["Yes", "No"])

# ---- Services ----
st.subheader("Services")
col3, col4 = st.columns(2)
with col3:
    phone = st.selectbox("Phone service", ["Yes", "No"])
    internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
with col4:
    # Some answers depend on others, so we fill them in automatically
    if phone == "No":
        multiple = "No phone service"
    else:
        multiple = st.selectbox("Multiple lines", ["No", "Yes"])

addon_names = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
               "TechSupport", "StreamingTV", "StreamingMovies"]
addons = {}
if internet == "No":
    # No internet means these add-ons can't exist
    for a in addon_names:
        addons[a] = "No internet service"
else:
    cols = st.columns(3)
    for i, a in enumerate(addon_names):
        with cols[i % 3]:
            addons[a] = st.selectbox(a, ["No", "Yes"])

# ---- Customer ----
st.subheader("Customer")
col5, col6 = st.columns(2)
with col5:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior citizen", ["No", "Yes"])
with col6:
    partner = st.selectbox("Has partner", ["Yes", "No"])
    dependents = st.selectbox("Has dependents", ["Yes", "No"])

# ---- Predict ----
if st.button("Check churn risk", type="primary"):
    row = {
        "gender": gender,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        **addons,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        # Total charges is estimated as tenure x monthly charges
        "TotalCharges": tenure * monthly,
    }
    customer = pd.DataFrame([row])

    # Probability that this customer churns
    prob = model.predict_proba(customer)[0, 1]

    st.subheader("Result")
    st.metric("Churn probability", f"{prob:.0%}")
    st.progress(float(prob))

    if prob >= cutoff:
        st.error(f"🔴 High risk: above the {cutoff:.0%} cutoff. Contact this customer first.")
    else:
        st.success(f"🟢 Low risk: below the {cutoff:.0%} cutoff.")

# ---- What the model relies on ----
st.divider()
st.subheader("What the model relies on most")
st.write(
    "We shuffled each column on customers the model had never seen and measured "
    "how much its score dropped. A longer bar means the model depends on that "
    "column more. This shows what the model uses, not why customers leave."
)

# This file was created once in Colab with permutation importance
importance_file = BASE / "importance.csv"
if importance_file.exists():
    drivers = pd.read_csv(importance_file)
    chart = alt.Chart(drivers).mark_bar().encode(
        x=alt.X("Importance:Q", title="Drop in model score when shuffled"),
        y=alt.Y("Clue:N", sort="-x", title=None),
    )
    st.altair_chart(chart, width="stretch")
else:
    st.info("importance.csv not found. Add it to the app folder to see this chart.")