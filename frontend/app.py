import streamlit as st
import pandas as pd
import requests
import plotly.express as px

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="AI Analytics Platform",
    layout="wide"
)

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------
st.title("🚀 AI Analytics Platform")

st.markdown(
    """
    Upload a dataset and run automated Machine Learning analysis.
    """
)

# ---------------------------------------------------
# API URL
# ---------------------------------------------------
API_URL = "https://ai-analytics-platform-1.onrender.com/analyze/"

# ---------------------------------------------------
# FILE UPLOAD
# ---------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload CSV or Excel File",
    type=["csv", "xlsx"]
)

# ---------------------------------------------------
# IF FILE EXISTS
# ---------------------------------------------------
if uploaded_file:

    try:

        if uploaded_file.name.endswith(".csv"):

            df = pd.read_csv(uploaded_file)

        else:

            df = pd.read_excel(uploaded_file)

        st.success("✅ Dataset Loaded Successfully")

    except Exception as e:

        st.error(f"File Error: {e}")
        st.stop()

    # ---------------------------------------------------
    # DATA PREVIEW
    # ---------------------------------------------------
    st.subheader("📊 Dataset Preview")

    col1, col2 = st.columns(2)

    with col1:

        st.write("Head")
        st.dataframe(
            df.head(),
            width='stretch'
        )

    with col2:

        st.write("Tail")
        st.dataframe(
            df.tail(),
            width='stretch'
        )

    # ---------------------------------------------------
    # DATASET INFO
    # ---------------------------------------------------
    st.subheader("📌 Dataset Information")

    a, b, c = st.columns(3)

    a.metric("Rows", df.shape[0])

    b.metric("Columns", df.shape[1])

    c.metric(
        "Missing Values",
        int(df.isnull().sum().sum())
    )

    # ---------------------------------------------------
    # TARGET
    # ---------------------------------------------------
    target = st.selectbox(
        "🎯 Select Target Column",
        df.columns
    )

    # ---------------------------------------------------
    # VISUALIZATION
    # ---------------------------------------------------
    st.subheader("📈 Visualization")

    numeric_columns = df.select_dtypes(
        include='number'
    ).columns

    if len(numeric_columns) > 0:

        chart = st.selectbox(
            "Select Chart",
            [
                "Histogram",
                "Scatter",
                "Box",
                "Line"
            ]
        )

        column = st.selectbox(
            "Select Column",
            numeric_columns
        )

        if chart == "Histogram":

            fig = px.histogram(
                df,
                x=column
            )

        elif chart == "Scatter":

            fig = px.scatter(
                df,
                x=numeric_columns[0],
                y=numeric_columns[-1],
                color=target
            )

        elif chart == "Box":

            fig = px.box(
                df,
                y=column,
                color=target
            )

        else:

            fig = px.line(
                df,
                y=column
            )

        st.plotly_chart(
            fig,
            width='stretch'
        )

    # ---------------------------------------------------
    # RUN ANALYSIS
    # ---------------------------------------------------
    if st.button("🚀 Run AI Analysis"):

        with st.spinner("Training ML Models..."):

            try:

                uploaded_file.seek(0)

                files = {

                    "file": (

                        uploaded_file.name,

                        uploaded_file.getvalue(),

                        uploaded_file.type

                    )
                }

                response = requests.post(
                    API_URL,
                    files=files,
                    params={"target": target},
                    timeout=300
                )

                st.write(
                    f"Status Code: {response.status_code}"
                )

                # ---------------------------------------
                # SUCCESS
                # ---------------------------------------
                if response.status_code == 200:

                    result = response.json()

                    if "error" in result:

                        st.error(result["error"])

                    else:

                        # -------------------------------
                        # RESULTS
                        # -------------------------------
                        st.subheader(
                            "🤖 AutoML Results"
                        )

                        st.json(result)

                        # -------------------------------
                        # METRICS
                        # -------------------------------
                        x1, x2, x3 = st.columns(3)

                        x1.metric(
                            "Task",
                            result.get("task", "N/A")
                        )

                        x2.metric(
                            "Best Model",
                            result.get(
                                "best_model",
                                "N/A"
                            )
                        )

                        x3.metric(
                            "Metric",
                            result.get(
                                "metric",
                                "N/A"
                            )
                        )

                        # -------------------------------
                        # SCORES
                        # -------------------------------
                        scores = result.get(
                            "scores",
                            {}
                        )

                        score_df = pd.DataFrame({

                            "Model":
                            list(scores.keys()),

                            "Score":
                            list(scores.values())

                        })

                        st.subheader(
                            "📊 Model Performance"
                        )

                        st.dataframe(
                            score_df,
                            width='stretch'
                        )

                        # -------------------------------
                        # BAR CHART
                        # -------------------------------
                        fig = px.bar(
                            score_df,
                            x="Model",
                            y="Score",
                            text="Score",
                            title="Model Comparison"
                        )

                        fig.update_traces(
                            textposition="outside"
                        )

                        st.plotly_chart(
                            fig,
                            width='stretch'
                        )

                        st.success(
                            f"""
                            🏆 Best Model:
                            {result['best_model']}
                            """
                        )

                else:

                    st.error(
                        f"""
                        Backend Error:
                        {response.status_code}
                        """
                    )

                    st.text(response.text)

            except requests.exceptions.Timeout:

                st.error(
                    """
                    Request Timeout.
                    Backend may be sleeping.
                    """
                )

            except Exception as e:

                st.error(f"Error: {e}")

else:

    st.info(
        "📂 Upload dataset to start analysis"
    )
