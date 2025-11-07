import os
from dotenv import load_dotenv
from openai import OpenAI

import streamlit as st
import pandas as pd
import numpy as np
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

st.title("🧠 Intelligent Business Insights Generator")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### ✅ Preview of Uploaded Data:")
    st.dataframe(df.head())

    st.write("---")
    st.write("## 🧼 Data Cleaning")

    # ----- DTYPE FIX -----
    df_clean = df.copy()
    for col in df_clean.columns:
        # Try convert to numeric
        df_clean[col] = pd.to_numeric(df_clean[col], errors="ignore")
        # Try convert to datetime
        df_clean[col] = pd.to_datetime(df_clean[col], errors="ignore")

    # ----- MISSING VALUES -----
    missing_before = df_clean.isnull().sum()
    df_clean = df_clean.fillna(df_clean.median(numeric_only=True))
    df_clean = df_clean.fillna("Unknown")
    missing_after = df_clean.isnull().sum()

    st.write("**Missing values before cleaning:**")
    st.write(missing_before[missing_before > 0])
    st.write("**Missing values after cleaning:**")
    st.write(missing_after[missing_after > 0])

    # ----- OUTLIERS (Clipping using IQR) -----
    numeric_cols = df_clean.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df_clean[col] = np.clip(df_clean[col], lower, upper)

    st.success("✅ Cleaning completed!")
    st.write("---")
    st.write("## 📊 Automated EDA")

    import matplotlib.pyplot as plt
    import seaborn as sns
    import io

    # -------- HISTOGRAMS --------
    st.write("### Distribution of Numeric Columns")
    for col in numeric_cols[:5]:  # limit to first 5
        fig, ax = plt.subplots()
        df_clean[col].hist(bins=30, ax=ax)
        ax.set_title(f"Histogram — {col}")
        st.pyplot(fig)

    # -------- CORRELATION HEATMAP --------
    if len(numeric_cols) > 1:
        st.write("### Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(6,4))
        sns.heatmap(df_clean[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    # -------- TIME TREND (IF ANY DATETIME COLUMN) --------
    date_cols = df_clean.select_dtypes(include=["datetime64[ns]"]).columns
    if len(date_cols) > 0:
        date_col = date_cols[0]  # pick first
        st.write(f"### Time Trend (using: {date_col})")
        df_sorted = df_clean.sort_values(by=date_col)
        for col in numeric_cols[:3]:
            fig, ax = plt.subplots()
            ax.plot(df_sorted[date_col], df_sorted[col])
            ax.set_title(f"{col} over time")
            st.pyplot(fig)
    st.write("---")
    st.write("## 🧠 Auto Insights & Recommendations")
    st.write("---")
    st.write("## 🧠 Auto Insights & Recommendations")

    # ---------- Stats summary to send to GPT ----------
    summary_text = f"""
    Rows: {df_clean.shape[0]}
    Columns: {df_clean.shape[1]}
    Numeric columns: {list(numeric_cols)}
    Sample head:
    {df_clean.head().to_string()}
    """

    insights_text = ""
    if OPENAI_API_KEY:
        try:
            client = OpenAI(api_key=OPENAI_API_KEY)
            prompt = f"""
            You are a senior data analyst. Read this dataset sample and produce
            6-10 clear English insights and 3-5 business action recommendations.

            DATA SUMMARY:
            {summary_text}
            """
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role":"user","content":prompt}],
                max_tokens=600,
                temperature=0.2,
            )
            insights_text = response.choices[0].message.content
        except Exception as e:
            insights_text = f"(GPT error fallback) {e}"

    else:
        insights_text = "⚠️ No API Key found — using rule-based insights.\n"
        insights_text += f"- Dataset has {df_clean.shape[0]} rows.\n"
        insights_text += "- Add OPENAI_API_KEY in .env for smarter insights.\n"

    st.text_area("Insights & Actions", value=insights_text, height=300)




