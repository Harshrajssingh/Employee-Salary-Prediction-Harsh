import streamlit as st
import joblib
import numpy as np

st.title("💼 Salary Prediction App")
st.divider()

st.write("📊 With this app, you can estimate the annual salary of company employees based on their experience and job performance.")

years = st.number_input("🧑‍💼 Enter years at company", value=1, step=1, min_value=0)
jobrate = st.number_input("📈 Enter the Job rate", value=3.5, step=0.5, min_value=0.0)

model = joblib.load("lineramodel.pkl")

st.divider()
predict = st.button("🔍 Predict Salary")

if predict:
    X = np.array([[years, jobrate]])
    prediction = model.predict(X)[0]

    st.balloons()
    st.success(f"💰 Estimated Annual Salary: ₹{prediction:,.2f}")
else:
    st.info("Press the button to get a salary prediction.")

st.divider()
st.caption("🧑‍💻 Created by HarshRaj Singh | 🧠 Powered by Streamlit + scikit-learn")
