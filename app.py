import streamlit as st
import pandas as pd
import joblib

# تحميل النموذج 
model = joblib.load("risk_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")


st.set_page_config(page_title="ICU Lab Alert System", layout="centered")
st.title("🧠 AI-Based ICU Alert System")
st.markdown("---")

lab_tests = ["Hemoglobin", "Hematocrit", "RBC", "Potassium"]
diagnoses = ["Post-operative bleeding", "DKA"]

diag = st.selectbox("Diagnosis", diagnoses)
test = st.selectbox("Lab Test", lab_tests)
result = st.number_input("Current Lab Result", format="%.2f")
previous = st.number_input("Previous Result (optional)", format="%.2f", value=0.0, step=0.1)
use_prev = st.checkbox("Use Previous Result", value=False)

CRITICAL_THRESHOLDS = {
    "Hemoglobin": lambda x: x < 7.0,
    "Hematocrit": lambda x: x < 21,
    "RBC": lambda x: x < 3.5,
    "Potassium": lambda x: x < 3.0 or x > 6.0
}

DIAGNOSIS_RULES = {
    "DKA": lambda t, x: t == "Potassium" and (x < 3.0 or x > 6.0),
    "Post-operative bleeding": lambda t, x: (t == "Hemoglobin" and x < 7.0) or (t == "Hematocrit" and x < 21)
}

if st.button("🔍 Analyze Result"):
    is_critical = CRITICAL_THRESHOLDS[test](result)
    is_diagnosis = DIAGNOSIS_RULES[diag](test, result)
    is_sudden = abs(previous - result) >= 5.0 if use_prev else False

    reasons = []
    if is_critical:
        reasons.append("🛑 Critical Level")
    if is_sudden:
        reasons.append("⚠️ Sudden Change")
    if is_diagnosis:
        reasons.append("⚠️ Diagnosis-based Risk")

    if reasons:
        st.error(f"🚨 **RISK Detected**\\n\\n**Reasons:** {', '.join(reasons)}")
    else:
        st.success("✅ No Risk detected based on current input.")


# زر التنبؤ الذكي
if st.button("🤖 Predict"):
    try:
        input_df = pd.DataFrame({
            "Diagnosis": [label_encoders["Diagnosis"].transform([diag])[0]],
            "Lab Test": [label_encoders["Lab Test"].transform([test])[0]],
            "Result": [result],
            "Previous Result": [previous]
        })

        prediction = model.predict(input_df)[0]
        if prediction == 1:
            st.error("🚨 Smart Model Prediction: RISK")
        else:
            st.success("✅ Smart Model Prediction: No Risk")
    except Exception as e:
        st.warning(f"⚠️ Prediction failed: {e}")

