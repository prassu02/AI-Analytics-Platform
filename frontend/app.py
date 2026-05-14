import streamlit as st
import pandas as pd
import requests
import plotly.express as px

st.set_page_config(layout="wide")

st.title("🚀 AI Analytics Platform (AutoML + Metrics)")

API_URL = "https://ai-analytics-platform-1.onrender.com/analyze/"

uploaded_file = st.file_uploader(
    "Upload CSV or Excel",
    type=["csv", "xlsx"]
)

if uploaded_file:

    # ----------------------------
    # LOAD DATA
    # ----------------------------
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    st.write("Rows:", df.shape[0])
    st.write("Columns:", df.shape[1])

    target = st.selectbox("Select Target Column", df.columns)

    # ----------------------------
    # RUN ANALYSIS
    # ----------------------------
    if st.button("🚀 Run AutoML"):

        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type
            )
        }

        with st.spinner("Running models..."):

            response = requests.post(
                API_URL,
                files=files,
                params={"target": target},
                timeout=300
            )

        # ----------------------------
        # SAFE JSON PARSE
        # ----------------------------
        try:
            result = response.json()
        except Exception:
            st.error("Backend did not return valid JSON")
            st.stop()

        if "error" in result:
            st.error(result["error"])
            st.stop()

        # ----------------------------
        # MAIN INFO
        # ----------------------------
        st.subheader("🤖 AutoML Results")

        st.success(f"🏆 Best Model: {result['best_model']}")
        st.info(f"📌 Task: {result['task']}")
        st.info(f"📊 Metric: {result['metric']}")
        st.write(f"Rows: {result['rows']} | Columns: {result['columns']}")

        # ----------------------------
        # SCORES TABLE
        # ----------------------------
        st.subheader("📈 Model Scores")

        score_df = pd.DataFrame({
            "Model": list(result["scores"].keys()),
            "Score": list(result["scores"].values())
        })

        st.dataframe(score_df, use_container_width=True)

        fig = px.bar(
            score_df,
            x="Model",
            y="Score",
            title="Model Performance Comparison"
        )
        st.plotly_chart(fig, use_container_width=True)

        # ----------------------------
        # ADVANCED METRICS
        # ----------------------------
        st.subheader("📊 Advanced Metrics (Per Model)")

        metrics = result.get("metrics", {})

        if metrics:

            for model_name, metric_values in metrics.items():

                st.markdown(f"### 🔹 {model_name}")

                metric_df = pd.DataFrame(
                    list(metric_values.items()),
                    columns=["Metric", "Value"]
                )

                st.dataframe(metric_df, use_container_width=True)

        else:
            st.warning("No detailed metrics returned from backend")

        # ----------------------------
        # BEST MODEL HIGHLIGHT CARD
        # ----------------------------
        st.subheader("🏆 Best Model Summary")

        best = result["best_model"]
        best_score = result["scores"][best]

        st.metric(
            label="Best Model",
            value=best,
            delta=f"Score: {best_score}"
        )
