import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.figure_factory as ff

st.set_page_config(
    page_title="AI Analytics Platform",
    layout="wide",
    page_icon="🚀"
)

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ AI Platform Controls")

API_URL = "https://ai-analytics-platform-1.onrender.com/analyze/"

uploaded_file = st.sidebar.file_uploader(
    "Upload Dataset",
    type=["csv", "xlsx"]
)

# ---------------- MAIN TITLE ----------------
st.title("🚀 AI Analytics Platform")
st.caption("AutoML | XAI | Metrics | Visualization Dashboard")

if uploaded_file:

    # Load dataset
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)

    # ---------------- TABS ----------------
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Dataset",
        "📈 Visualization",
        "🤖 AutoML",
        "📊 Metrics"
    ])

    # ================= DATASET =================
    with tab1:

        st.subheader("Dataset Overview")

        col1, col2, col3 = st.columns(3)

        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])
        col3.metric("Missing Values", df.isnull().sum().sum())

        st.dataframe(df.head(), use_container_width=True)

        st.subheader("Statistical Summary")
        st.dataframe(df.describe(include="all"), use_container_width=True)

    # ================= VISUALIZATION =================
    with tab2:

        st.subheader("Data Visualization")

        numeric_cols = df.select_dtypes(include="number").columns

        chart = st.selectbox(
            "Select Chart",
            ["Histogram", "Scatter", "Box", "Correlation Heatmap"]
        )

        if len(numeric_cols) > 0:

            if chart == "Histogram":
                col = st.selectbox("Column", numeric_cols)
                fig = px.histogram(df, x=col)
                st.plotly_chart(fig, use_container_width=True)

            elif chart == "Scatter":
                x = st.selectbox("X-axis", numeric_cols)
                y = st.selectbox("Y-axis", numeric_cols)
                fig = px.scatter(df, x=x, y=y)
                st.plotly_chart(fig, use_container_width=True)

            elif chart == "Box":
                col = st.selectbox("Column", numeric_cols)
                fig = px.box(df, y=col)
                st.plotly_chart(fig, use_container_width=True)

            elif chart == "Correlation Heatmap":
                fig = ff.create_annotated_heatmap(
                    z=df[numeric_cols].corr().values,
                    x=list(numeric_cols),
                    y=list(numeric_cols),
                    colorscale="Viridis"
                )
                st.plotly_chart(fig, use_container_width=True)

    # ================= AUTO ML =================
    with tab3:

        st.subheader("AutoML Engine")

        target = st.selectbox("Select Target Column", df.columns)

        if st.button("🚀 Run AutoML"):

            files = {
                "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
            }

            response = requests.post(
                API_URL,
                files=files,
                params={"target": target}
            )

            try:
                result = response.json()
            except:
                st.error("Backend error")
                st.stop()

            if "error" in result:
                st.error(result["error"])
                st.stop()

            st.success("Model Training Completed!")

            st.subheader("🏆 Best Model")
            st.metric("Best Model", result["best_model"], result["scores"][result["best_model"]])

            score_df = pd.DataFrame({
                "Model": list(result["scores"].keys()),
                "Score": list(result["scores"].values())
            })

            fig = px.bar(
                score_df,
                x="Model",
                y="Score",
                color="Score",
                text="Score",
                title="Model Comparison"
            )

            st.plotly_chart(fig, use_container_width=True)

    # ================= METRICS =================
    with tab4:

        st.subheader("Advanced Model Metrics")

        if "result" in locals():

            metrics = result.get("metrics", {})

            for model, values in metrics.items():

                st.markdown(f"### 🔹 {model}")

                col1, col2, col3, col4 = st.columns(4)

                for i, (k, v) in enumerate(values.items()):

                    if i == 0:
                        col1.metric(k, v)
                    elif i == 1:
                        col2.metric(k, v)
                    elif i == 2:
                        col3.metric(k, v)
                    elif i == 3:
                        col4.metric(k, v)

                st.divider()

else:
    st.info("👈 Upload dataset from sidebar to start")
