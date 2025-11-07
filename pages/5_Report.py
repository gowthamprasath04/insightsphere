import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont

# ---------- PAGE CONFIG ----------
st.title("📄 InsightSphere — Executive Report Generator")

st.write("Generate a professional, print-ready PDF report summarizing data insights and ML results.")

# Client name input
client_name = st.text_input("Enter Client Name (optional):", "")
generate_btn = st.button("🧾 Generate Executive PDF Report")

# ---------- VALIDATION ----------
if generate_btn:

    if "df_clean" not in st.session_state:
        st.error("No cleaned dataset found. Please complete earlier steps first.")
        st.stop()

    df = st.session_state["df_clean"]
    ml_report = st.session_state.get("ml_report", "No ML report available.")

    # ---------- PDF BUFFER ----------
    buffer = BytesIO()
    pdf = SimpleDocTemplate(buffer, pagesize=A4)
    pdfmetrics.registerFont(UnicodeCIDFont("HeiseiMin-W3"))  # Serif font (F2)

    # ---------- STYLES ----------
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="CenterTitle", fontName="HeiseiMin-W3",
                              alignment=TA_CENTER, fontSize=16, spaceAfter=12))
    styles.add(ParagraphStyle(name="SubHeader", fontName="HeiseiMin-W3",
                              alignment=TA_LEFT, fontSize=13, spaceAfter=6))
    styles.add(ParagraphStyle(name="NormalText", fontName="HeiseiMin-W3",
                              alignment=TA_LEFT, fontSize=11, leading=14))

    story = []

    # ---------- TITLE PAGE ----------
    title_text = "<b>InsightSphere — Unified Analytics & Intelligence Platform</b>"
    story.append(Paragraph(title_text, styles["CenterTitle"]))

    if client_name:
        story.append(Paragraph(f"<b>Client:</b> {client_name}", styles["NormalText"]))
    story.append(Paragraph("<b>Analyst:</b> Gowtham Prasath", styles["NormalText"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("<b>Report Type:</b> Executive Summary", styles["NormalText"]))
    story.append(Paragraph("<b>Theme:</b> Black & White — Portrait Layout", styles["NormalText"]))
    story.append(Spacer(1, 24))

    # ---------- DATA SUMMARY ----------
    story.append(Paragraph("1️⃣ Dataset Summary", styles["SubHeader"]))
    story.append(Paragraph(f"Rows: <b>{df.shape[0]}</b> Columns: <b>{df.shape[1]}</b>", styles["NormalText"]))
    story.append(Spacer(1, 6))

    # Limit to first 5 columns for clean layout
    cols_to_show = df.columns[:5]
    story.append(Paragraph(
        f"Showing first 5 columns out of {df.shape[1]} total for readability.",
        styles["NormalText"])
    )

    # Create sample data table
    sample_table = Table([cols_to_show.tolist()] + df[cols_to_show].head(5).values.tolist())
    sample_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.black),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.black),
        ("FONTNAME", (0, 0), (-1, -1), "HeiseiMin-W3"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
    ]))
    story.append(sample_table)
    story.append(Spacer(1, 18))

    # ---------- AUTO INSIGHTS ----------
    if "auto_insights" in st.session_state:
        story.append(Paragraph("2️⃣ AI-Generated Insights", styles["SubHeader"]))
        story.append(Paragraph(st.session_state["auto_insights"], styles["NormalText"]))
        story.append(Spacer(1, 18))

    # ---------- ML EVALUATION ----------
    story.append(Paragraph("3️⃣ Machine Learning Evaluation", styles["SubHeader"]))
    story.append(Paragraph("Classification Report:", styles["NormalText"]))
    story.append(Paragraph(f"<pre>{ml_report}</pre>", styles["NormalText"]))
    story.append(Spacer(1, 12))

    # ---------- CHART EXPORTS ----------
    if "roc_curve" in st.session_state:
        roc_fig = st.session_state["roc_curve"]
        buf = BytesIO()
        roc_fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        story.append(Image(buf, width=450, height=250))
        story.append(Spacer(1, 12))

    if "pr_curve" in st.session_state:
        pr_fig = st.session_state["pr_curve"]
        buf = BytesIO()
        pr_fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        story.append(Image(buf, width=450, height=250))
        story.append(Spacer(1, 12))

    if "feature_importance" in st.session_state:
        fi_fig = st.session_state["feature_importance"]
        buf = BytesIO()
        fi_fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        story.append(Image(buf, width=450, height=250))
        story.append(Spacer(1, 12))

    # ---------- BUSINESS RECOMMENDATIONS ----------
    story.append(Paragraph("4️⃣ Business Recommendations", styles["SubHeader"]))
    rec_text = (
        "• Focus marketing on high-conversion segments.<br/>"
        "• Address churn drivers identified by feature importance.<br/>"
        "• Maintain data quality for predictive reliability.<br/>"
        "• Integrate InsightSphere pipeline for continuous monitoring."
    )
    story.append(Paragraph(rec_text, styles["NormalText"]))
    story.append(Spacer(1, 24))

    story.append(Paragraph("End of Report", styles["CenterTitle"]))

    # ---------- BUILD PDF ----------
    pdf.build(story)

    pdf_value = buffer.getvalue()
    buffer.close()

    # ---------- DOWNLOAD ----------
    filename = f"InsightSphere_Report_{client_name if client_name else 'Client'}.pdf"
    st.download_button(
        label="⬇️ Download PDF Report",
        data=pdf_value,
        file_name=filename,
        mime="application/pdf",
    )
    st.success("✅ Executive PDF Report Generated Successfully!")
