import streamlit as st

st.set_page_config(
    page_title="InsightSphere",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
    <style>
    .main {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        color: #f1f5f9;
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3 {
        color: #38bdf8;
        font-weight: 700;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🌐 Welcome to InsightSphere")

st.markdown("""
### 🧩 What You Can Do Here:
1. **Upload** your dataset (CSV format)
2. **Clean & Explore** it automatically
3. **Generate AI Insights** without needing API keys
4. **Run ML-based Churn Prediction**
5. **Export a Professional PDF Report**

Use the sidebar to navigate between sections 👉
""")

st.markdown("---")
st.markdown("#### 👨‍💻 Developed by **Gowtham Prasath** — Data Analyst at InsightSphere")
