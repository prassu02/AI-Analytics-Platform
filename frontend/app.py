import streamlit as st
import pandas as pd
import requests
import plotly.express as px

st.set_page_config(layout="wide")

st.title("🚀 AI Analytics Platform")

API_URL = "https://ai-analytics-platform-1.onrender.com/analyze/"

file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

if file:

    df = pd.read_csv(file) if file.name.endswith("csv") else pd.read_excel(file)

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    target = st.selectbox("Select Target Column", df.columns)

    if st.button("🚀 Run AutoML"):

        files = {
            "file": (file.name, file.getvalue(), file.type)
        }

        res = requests.post(API_URL, files=files, params={"target": target})

        try:
            result = res.json()
        except:
            st.error("Backend error: No JSON response")
            st.stop()

        if "error" in result:
            st.error(result["error"])
            st.stop()

        # ---------------- RESULTS ----------------
        st.subheader("🤖 AutoML Results")

        st.success(f"Best Model: {result['best_model']}")
        st.info(f"Task: {result['task']}")
        st.info(f"Metric: {result['metric']}")

        # ---------------- SCORES ----------------
        score_df = pd.DataFrame({
            "Model": list(result["scores"].keys()),
            "Score": list(result["scores"].values())
        })

        st.subheader("📈 Model Scores")
        st.dataframe(score_df, use_container_width=True)

        fig = px.bar(score_df, x="Model", y="Score", title="Model Comparison")
        st.plotly_chart(fig, use_container_width=True)

        # ---------------- METRICS ----------------
        st.subheader("📊 Detailed Metrics")

        metrics = result.get("metrics", {})

        for model, values in metrics.items():

            st.markdown(f"### 🔹 {model}")

            df_metrics = pd.DataFrame(
                list(values.items()),
                columns=["Metric", "Value"]
            )

            st.dataframe(df_metrics, use_container_width=True)

        # ---------------- BEST MODEL CARD ----------------
        best = result["best_model"]
        st.metric("🏆 Best Model", best, result["scores"][best])
