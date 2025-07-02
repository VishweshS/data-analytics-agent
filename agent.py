# to run: streamlit run agent.py
# import all the required libraries
import io
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import streamlit as st
import google.generativeai as genai

import warnings
warnings.filterwarnings('ignore')

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

st.set_page_config(page_title="Agentic Data Explorer", layout="wide")
st.title("Explore Your Data — Descriptive Insights")

uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    # Read file
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
	
	# --- Basic Exploration ---
    st.subheader("📌 Data Preview")
    st.dataframe(df.head())

    st.subheader("🔎 Basic Statistics")
    st.dataframe(df.describe())

    st.subheader("📋 Column Info")
    buffer = io.StringIO()
    df.info(buf=buffer)
    st.text(buffer.getvalue())

    # --- Auto Visualizations ---
    st.subheader("📊 Auto Visualizations (Top 2 Columns)")

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if numeric_cols:
        col = numeric_cols[0]
        st.markdown(f"### Histogram for `{col}`")
        fig, ax = plt.subplots()
        sns.histplot(df[col].dropna(), kde=True, ax=ax)
        st.pyplot(fig)

    if categorical_cols:
        col = categorical_cols[0] # Added this line to define 'col' for categorical plots

        # Limiting to the top 15 categories by count
        top_cats = df[col].value_counts().head(15)

        st.markdown(f"### Top 15 Categories in `{col}`")
        fig, ax = plt.subplots()
        top_cats.plot(kind='bar', ax=ax)
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)
		
    st.divider()
    st.subheader("🧠 Ask Anything About Your Data")
    prompt = st.text_input("Try: 'Which category has the highest average sales?'")

    if prompt:
        # Convert data to string to give Gemini context
        sample_data = df.head(20).to_string(index=False)
        full_prompt = f"""
You are a data analyst. Here is a preview of the dataset:

{sample_data}

Question: {prompt}
"""

        with st.spinner("Gemini is thinking..."):
            try:
                model = genai.GenerativeModel("gemini-2.5-pro")
                response = model.generate_content(full_prompt)
                st.success("✅ Answer:")
                st.markdown(f"> {response.text}")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")