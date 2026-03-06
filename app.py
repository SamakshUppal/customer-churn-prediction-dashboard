import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")

st.title("📊 Customer Churn Prediction Dashboard")

st.write(
"This dashboard analyzes telecom customer behavior and predicts whether a customer is likely to churn."
)

# Load model
model = pickle.load(open("model.pkl","rb"))
scaler = pickle.load(open("scaler.pkl","rb"))

# Load dataset
df = pd.read_csv("data/churn.csv")
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna()

# Sidebar
st.sidebar.header("Customer Input")

tenure = st.sidebar.number_input("Tenure (Months)",0)
monthly_charges = st.sidebar.number_input("Monthly Charges",0.0)
total_charges = st.sidebar.number_input("Total Charges",0.0)

# Prediction
if st.sidebar.button("Predict Churn"):

    input_data = np.array([[tenure, monthly_charges, total_charges]])
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.sidebar.error("Customer likely to churn")
    else:
        st.sidebar.success("Customer likely to stay")

st.markdown("---")

# Dashboard Charts
st.subheader("Dataset Insights")

col1, col2 = st.columns(2)

with col1:

    st.write("Churn Distribution")

    fig, ax = plt.subplots()
    sns.countplot(x="Churn", data=df, ax=ax)
    st.pyplot(fig)

with col2:

    st.write("Monthly Charges vs Churn")

    fig, ax = plt.subplots()
    sns.boxplot(x="Churn", y="MonthlyCharges", data=df, ax=ax)
    st.pyplot(fig)

st.markdown("---")

st.subheader("Tenure Distribution")

fig, ax = plt.subplots()
sns.histplot(df["tenure"], bins=30, kde=True, ax=ax)
st.pyplot(fig)

st.markdown("---")

st.write("### Project Information")

st.write("""
• Model: Random Forest Classifier  
• Dataset: IBM Telco Customer Churn Dataset  
• Features Used: Tenure, MonthlyCharges, TotalCharges  
• Libraries: Pandas, Scikit-learn, Streamlit, Matplotlib, Seaborn
""")