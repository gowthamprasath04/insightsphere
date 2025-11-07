import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.title("🧼 Data Cleaning & 📊 Exploratory Data Analysis")

# Ensure dataset exists
if "df" not in st.session_state:
    st.warning("⚠️ No data found. Please upload a CSV from the 'Upload Data' page.")
    st.stop()

df = st.session_state["df"].copy()

st.subheader("1) Initial Data Overview")
st.write(df.head())
st.write(f"**Rows:** {df.shape[0]}, **Columns:** {df.shape[1]}")
st.write("**Columns:**", list(df.columns))

# ---------- CLEANING ----------
st.subheader("2) Data Cleaning")

# Convert dtypes
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="ignore")
    df[col] = pd.to_datetime(df[col], errors="ignore")

# Handle missing values
missing_before = df.isnull().sum()
df = df.fillna(df.median(numeric_only=True))
df = df.fillna("Unknown")
missing_after = df.isnull().sum()

st.write("**Missing Values (Before):**")
st.write(missing_before[missing_before > 0])
st.write("**Missing Values (After):**")
st.write(missing_after[missing_after > 0])

# Handle outliers
num_cols = df.select_dtypes(include=["number"]).columns
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5*IQR
    upper = Q3 + 1.5*IQR
    df[col] = np.clip(df[col], lower, upper)

# Save cleaned df for next pages
st.session_state["df_clean"] = df

st.success("✅ Cleaning Completed")

# ---------- EDA ----------
st.subheader("3) Exploratory Data Analysis")

# Histograms
st.write("### Distribution of Numeric Columns")
for col in num_cols[:5]:
    fig, ax = plt.subplots()
    df[col].hist(bins=30, ax=ax)
    ax.set_title(f"Histogram — {col}")
    st.pyplot(fig)

# Correlation Heatmap
if len(num_cols) > 1:
    st.write("### Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
else:
    st.info("Not enough numeric columns for correlation heatmap.")

# Time Trends
date_cols = df.select_dtypes(include=["datetime64[ns]"]).columns
if len(date_cols) > 0:
    st.write(f"### Time Trend (using: {date_cols[0]})")
    df_sorted = df.sort_values(by=date_cols[0])
    for col in num_cols[:3]:
        fig, ax = plt.subplots()
        ax.plot(df_sorted[date_cols[0]], df_sorted[col])
        ax.set_title(f"{col} over time")
        st.pyplot(fig)
else:
    st.info("No date column detected — skipping time trend.")
