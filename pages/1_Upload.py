import streamlit as st
import pandas as pd

st.title("📤 Upload Data")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### ✅ Preview of Uploaded Data:")
    st.dataframe(df.head())

    # Store in session so other pages can use it
    st.session_state["df"] = df
    st.success("Dataset is loaded and saved for this session.")

    with st.expander("Dataset info"):
        st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
        st.write("Columns:", list(df.columns))

    if st.button("Clear uploaded data"):
        st.session_state.pop("df", None)
        st.experimental_rerun()
else:
    st.info("Please upload a CSV to begin.")
