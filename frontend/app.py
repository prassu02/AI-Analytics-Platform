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
    Upload a dataset and run automated Machine Learning analysis
    using FastAPI + Streamlit.
    """
)

# ---------------------------------------------------
# BACKEND API
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

    # ---------------------------------------------------
    # READ FILE
    # ---------------------------------------------------
    try:

        if uploaded_file.name.endswith(".csv"):

            df = pd.read_csv(uploaded_file)

        else:

            df = pd.read_excel(uploaded_file)

        st.success("✅ Dataset Loaded Successfully")

    except Exception as e:

        st.error(f"File Reading Error: {e}")
        st.stop()

    # ---------------------------------------------------
    # DATASET PREVIEW
    # ---------------------------------------------------
    st.subheader("📊 Dataset Preview")

    col1, col2 = st.columns(2)

    with col1:

        st.write("### Head")
        st.dataframe(df.head(), width='stretch')

    with col2:

        st.write("### Tail")
        st.dataframe(df.tail(), width='stretch')

    # ---------------------------------------------------
    # DATASET INFO
    # ---------------------------------------------------
    st.subheader("📌 Dataset Information")

    info1, info2, info3 = st.columns(3)

    info1.metric("Rows", df.shape[0])
    info2.metric("Columns", df.shape[1])
    info3.metric(
        "Missing Values",
        int(df.isnull().sum().sum())
    )

    # ---------------------------------------------------
    # TARGET SELECTION
    # ---------------------------------------------------
    st.subheader("🎯 Target Column")

    target = st.selectbox(
        "Select Target Variable",
        df.columns
    )

    # ---------------------------------------------------
    # VISUALIZATION
    # ---------------------------------------------------
    st.subheader("📈 Data Visualization")

    chart = st.selectbox(
        "Choose Visualization",
        [
            "Histogram",
            "Scatter",
            "Box",
            "Line",
            "Bar"
        ]
    )

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns

    if len(numeric_columns) > 0:

        column = st.selectbox(
            "Select Numeric Column",
            numeric_columns
        )

        # -----------------------------------------------
        # CHARTS
        # -----------------------------------------------
        if chart == "Histogram":

            fig = px.histogram(
                df,
                x=column,
                title=f"{column} Distribution"
            )

        elif chart == "Scatter":

            fig = px.scatter(
                df,
                x=numeric_columns[0],
                y=numeric_columns[-1],
                color=target,
                title="Scatter Plot"
            )

        elif chart == "Box":

            fig = px.box(
                df,
                y=column,
                color=target,
                title="Box Plot"
            )

        elif chart == "Line":

            fig = px.line(
                df,
                y=column,
                title="Line Plot"
            )

        else:

            fig = px.bar(
                df,
                x=df.index,
                y=column,
                title="Bar Chart"
            )

        st.plotly_chart(
            fig,
            width='stretch'
        )

    # ---------------------------------------------------
    # RUN ANALYSIS
    # ---------------------------------------------------
    st.subheader("🤖 AutoML Analysis")

    if st.button("Run AI Analysis"):

        with st.spinner(
            "Running Machine Learning Models..."
        ):

            try:

                # ---------------------------------------
                # RESET FILE POINTER
                # ---------------------------------------
                uploaded_file.seek(0)

                # ---------------------------------------
                # FILES
                # ---------------------------------------
                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type
                    )
                }

                # ---------------------------------------
                # API REQUEST
                # ---------------------------------------
                response = requests.post(
                    API_URL,
                    files=files,
                    params={"target": target},
                    timeout=300
                )

                # ---------------------------------------
                # STATUS
                # ---------------------------------------
                st.write(
                    f"✅ Status Code: {response.status_code}"
                )

                # ---------------------------------------
                # SUCCESS RESPONSE
                # ---------------------------------------
                if response.status_code == 200:

                    result = response.json()

                    # -----------------------------------
                    # ERROR HANDLING
                    # -----------------------------------
                    if "error" in result:

                        st.error(result["error"])

                    else:

                        # -------------------------------
                        # RESULT SECTION
                        # -------------------------------
                        st.subheader(
                            "🤖 AutoML Results"
                        )

                        st.json(result)

                        # -------------------------------
                        # TASK + BEST MODEL
                        # -------------------------------
                        c1, c2, c3 = st.columns(3)

                        c1.metric(
                            "Task",
                            result.get("task", "N/A")
                        )

                        c2.metric(
                            "Best Model",
                            result.get(
                                "best_model",
                                "N/A"
                            )
                        )

                        c3.metric(
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

                        if scores:

                            score_df = pd.DataFrame({

                                "Model": list(scores.keys()),

                                "Score": list(scores.values())

                            })

                            st.subheader(
                                "📊 Model Performance"
                            )

                            st.dataframe(
                                score_df,
                                width='stretch'
                            )

                            # ---------------------------
                            # SCORE CHART
                            # ---------------------------
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

                            # ---------------------------
                            # BEST MODEL SUCCESS
                            # ---------------------------
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
                    Render free tier may be sleeping.
                    """
                )

            except requests.exceptions.JSONDecodeError:

                st.error(
                    """
                    Invalid JSON response from backend.
                    """
                )

                st.text(response.text)

            except Exception as e:

                st.error(f"Unexpected Error: {e}")

# ---------------------------------------------------
# NO FILE
# ---------------------------------------------------
else:

    st.info("📂 Upload dataset to begin analysis")
