import streamlit as st
import pandas as pd
import random
import numpy as np
from sklearn.preprocessing import LabelEncoder

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="AI Insights", page_icon="🧠", layout="wide")
st.title("🧠 AI Insights & Recommendations (Smart Offline Mode)")

# -------------------------------
# LOAD DATA
# -------------------------------
if "df" not in st.session_state:
    st.warning("⚠️ Please upload a dataset first from the 'Upload' page.")
    st.stop()

df = st.session_state["df"]

# -------------------------------
# SUMMARY INFO
# -------------------------------
st.markdown(f"✅ **Dataset Summary:** {df.shape[0]} rows × {df.shape[1]} columns")

numeric_cols = list(df.select_dtypes(include=["number"]).columns)
categorical_cols = list(df.select_dtypes(exclude=["number"]).columns)

# -------------------------------
# OFFLINE RULE-BASED INSIGHTS ENGINE
# -------------------------------
def generate_rule_based_insights(df):
    insights = []
    recs = []

    # ---- Basic stats
    missing = df.isnull().sum().sum()
    if missing > 0:
        insights.append(f"The dataset contains {missing} missing values that may affect analysis.")
        recs.append("Handle missing values using imputation or removal before modeling.")
    else:
        insights.append("No missing values found — the dataset is clean and ready for analysis.")
        recs.append("Proceed with feature exploration or modeling directly.")

    # ---- Numeric trends
    if len(numeric_cols) > 0:
        num_df = df[numeric_cols].select_dtypes(include=[np.number])
        means = num_df.mean().sort_values(ascending=False)
        top = means.index[0]
        insights.append(f"'{top}' has the highest average value ({means.iloc[0]:.2f}) among numeric features.")
        recs.append(f"Investigate '{top}' further — it may be a key performance driver.")
    else:
        insights.append("No numeric columns detected in the dataset.")
        recs.append("Add or derive numeric indicators to enable quantitative analysis.")

    # ---- Categorical patterns
    if len(categorical_cols) > 0:
        top_cat = categorical_cols[0]
        top_val = df[top_cat].value_counts().idxmax()
        insights.append(f"In '{top_cat}', '{top_val}' appears most frequently — indicating a dominant category.")
        recs.append(f"Balance category '{top_val}' if imbalance causes bias, or target it in marketing strategies.")
    else:
        insights.append("No categorical columns found in this dataset.")
        recs.append("Include categorical fields for demographic or segment analysis.")

    # ---- Correlation analysis
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr().abs()
        corr_pairs = corr.unstack().sort_values(ascending=False)
        for (col1, col2), val in corr_pairs.items():
            if col1 != col2 and val > 0.7:
                insights.append(f"Strong correlation ({val:.2f}) between '{col1}' and '{col2}'.")
                recs.append(f"Consider removing one to avoid redundancy in modeling.")
                break
    else:
        insights.append("Not enough numeric data to compute meaningful correlations.")
        recs.append("Add more numerical metrics for deeper statistical insight.")

    # ---- Add variety
    random.shuffle(insights)
    random.shuffle(recs)

    return insights[:3], recs[:3]


# -------------------------------
# RUN ENGINE
# -------------------------------
if st.button("🔍 Generate AI Insights"):
    with st.spinner("🧠 Generating offline insights..."):
        insights, recs = generate_rule_based_insights(df)

    st.markdown("### 📊 **Top Insights:**")
    for i, text in enumerate(insights, 1):
        st.success(f"🟢 Insight {i}: {text}")

    st.markdown("### 💡 **Recommendations:**")
    for i, text in enumerate(recs, 1):
        st.info(f"🔵 Action {i}: {text}")

    # Store for report
    st.session_state["ai_insights"] = "\n".join(insights + recs)
else:
    st.info("Click **'Generate AI Insights'** to analyze trends instantly (no API key needed).")
