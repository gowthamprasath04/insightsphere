import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns

st.title("🤖 ML — Churn Prediction")

# Ensure data exists
if "df_clean" not in st.session_state:
    st.warning("⚠️ No cleaned data found. Go to 'Data Cleaning & EDA' first.")
    st.stop()

df = st.session_state["df_clean"].copy()

# ---------- STEP 1: Detect / Create Churn ----------
st.header("1) Detecting Churn Column")

churn_cols = [c for c in df.columns if "churn" in c.lower()]

if churn_cols:
    target_col = churn_cols[0]
    st.success(f"✅ Using existing churn column: **{target_col}**")
else:
    st.warning("No churn column found — creating synthetic churn.")
    num_df = df.select_dtypes(include=['number'])
    if num_df.shape[1] == 0:
        df["churn"] = np.random.randint(0, 2, size=len(df))
        st.info("No numeric columns → random churn created.")
    else:
        first_col = num_df.iloc[:, 0]
        df["churn"] = (first_col > first_col.median()).astype(int)
        st.info(f"Churn created using {num_df.columns[0]} median split.")
    target_col = "churn"

y = df[target_col]

# ---------- STEP 2: Prepare Features ----------
st.header("2) Preparing Features")

X = df.drop(columns=[target_col], errors="ignore")
date_cols = X.select_dtypes(include=["datetime64[ns]"]).columns
X = X.drop(columns=date_cols)

cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = X.select_dtypes(include=["number"]).columns.tolist()

for c in cat_cols:
    X[c] = X[c].astype(str)

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols)
    ]
)

# ---------- STEP 3: Train/Test Split & Train Model ----------
st.header("3) Training Model")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Pipeline(steps=[
    ("prep", preprocessor),
    ("clf", RandomForestClassifier(random_state=42))
])

model.fit(X_train, y_train)
pred = model.predict(X_test)

st.success("✅ Model Training Complete")

# ---------- STEP 4: Evaluation ----------
st.header("4) Model Evaluation")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Confusion Matrix")
    st.write(confusion_matrix(y_test, pred))

with col2:
    st.subheader("Classification Report")
    st.text(classification_report(y_test, pred))

# ---------- ROC & PRECISION-RECALL ----------
st.header("5) ROC & Precision–Recall Curves")

proba = model.predict_proba(X_test)[:, 1]

# ROC curve
fpr, tpr, _ = roc_curve(y_test, proba)
roc_auc = auc(fpr, tpr)

# PR curve
precision, recall, _ = precision_recall_curve(y_test, proba)

col3, col4 = st.columns(2)

with col3:
    st.subheader("ROC Curve (AUC = {:.2f})".format(roc_auc))
    plt.figure(figsize=(4,3))
    sns.lineplot(x=fpr, y=tpr)
    plt.plot([0,1],[0,1],'--',color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    st.pyplot(plt.gcf())
    plt.close()

with col4:
    st.subheader("Precision–Recall Curve")
    plt.figure(figsize=(4,3))
    sns.lineplot(x=recall, y=precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    st.pyplot(plt.gcf())
    plt.close()

# ---------- Feature Importance ----------
st.header("6) Feature Importance (Top 10)")

# Extract feature names after one-hot encoding
ohe = model.named_steps["prep"].named_transformers_["cat"]
encoded_cols = ohe.get_feature_names_out(cat_cols)
all_features = list(encoded_cols) + num_cols

importances = model.named_steps["clf"].feature_importances_
fi = pd.DataFrame({"feature": all_features, "importance": importances})
fi = fi.sort_values("importance", ascending=False).head(10)
fi["importance"] = fi["importance"] * 100  # convert to percentages

plt.figure(figsize=(6,4))
sns.barplot(x="importance", y="feature", data=fi, orient='h')
plt.xlabel("Importance (%)")
plt.ylabel("Feature")
plt.title("Top 10 Features Driving Churn")
st.pyplot(plt.gcf())
plt.close()

st.success("🎯 ML analysis completed and visualized.")
# ---------- SAVE RESULTS FOR REPORT PAGE ----------
st.session_state["ml_report"] = classification_report(y_test, pred)
st.session_state["roc_curve"] = plt.figure()
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
st.session_state["roc_curve"] = plt.gcf()
plt.close()

st.session_state["pr_curve"] = plt.figure()
plt.figure(figsize=(6, 4))
plt.plot(recall, precision, label="Precision–Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve")
plt.legend()
st.session_state["pr_curve"] = plt.gcf()
plt.close()

st.session_state["feature_importance"] = plt.figure()
plt.figure(figsize=(6, 4))
sns.barplot(x="importance", y="feature", data=fi, orient="h")
plt.xlabel("Importance (%)")
plt.ylabel("Feature")
plt.title("Top 10 Features Driving Churn")
st.session_state["feature_importance"] = plt.gcf()
plt.close()
