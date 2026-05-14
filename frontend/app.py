import streamlit as st
import pandas as pd
import requests
import plotly.express as px

st.set_page_config(layout='wide')

st.title('🚀 AI Analytics Platform')

API_URL = "https://ai-analytics-platform-1.onrender.com/analyze/"

uploaded_file = st.file_uploader(
    'Upload CSV or Excel',
    type=['csv', 'xlsx']
)

if uploaded_file:

    # -----------------------------
    # READ DATASET
    # -----------------------------
    try:

        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)

        else:
            df = pd.read_excel(uploaded_file)

        st.success('Dataset Loaded Successfully')

    except Exception as e:

        st.error(f"File Reading Error: {e}")
        st.stop()

    # -----------------------------
    # PREVIEW
    # -----------------------------
    st.subheader('📊 Dataset Preview')

    st.dataframe(df.head())

    st.write('Shape:', df.shape)

    # -----------------------------
    # TARGET
    # -----------------------------
    target = st.selectbox(
        'Select Target Column',
        df.columns
    )

    # -----------------------------
    # VISUALIZATION
    # -----------------------------
    chart = st.selectbox(
        'Visualization',
        ['Histogram', 'Scatter', 'Box', 'Line']
    )

    numeric_columns = df.select_dtypes(include='number').columns

    if len(numeric_columns) > 0:

        column = st.selectbox(
            'Column',
            numeric_columns
        )

        if chart == 'Histogram':

            fig = px.histogram(df, x=column)

        elif chart == 'Scatter':

            fig = px.scatter(
                df,
                x=numeric_columns[0],
                y=numeric_columns[-1]
            )

        elif chart == 'Box':

            fig = px.box(df, y=column)

        else:

            fig = px.line(df, y=column)

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # -----------------------------
    # AI ANALYSIS
    # -----------------------------
    if st.button('Run AI Analysis'):

        with st.spinner("Running AutoML Analysis..."):

            try:

                # Reset file pointer
                uploaded_file.seek(0)

                files = {
                    'file': (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type
                    )
                }

                response = requests.post(
                    API_URL,
                    files=files,
                    params={'target': target},
                    timeout=300
                )

                # -----------------------------
                # DEBUG STATUS
                # -----------------------------
                st.write("Status Code:", response.status_code)

                # -----------------------------
                # HANDLE SUCCESS
                # -----------------------------
                if response.status_code == 200:

                    result = response.json()

                    st.subheader('🤖 AutoML Results')

                    st.json(result)

                    # -----------------------------
                    # SCORES
                    # -----------------------------
                    scores = result.get('scores', {})

                    if scores:

                        score_df = pd.DataFrame({
                            'Model': list(scores.keys()),
                            'Score': list(scores.values())
                        })

                        st.dataframe(score_df)

                        fig = px.bar(
                            score_df,
                            x='Model',
                            y='Score'
                        )

                        st.plotly_chart(
                            fig,
                            use_container_width=True
                        )

                    st.success(
                        f"Best Model: {result.get('best_model', 'N/A')}"
                    )

                else:

                    st.error(
                        f"Backend Error {response.status_code}"
                    )

                    st.text(response.text)

            except requests.exceptions.Timeout:

                st.error(
                    "Request Timeout. Render free tier may be sleeping."
                )

            except requests.exceptions.JSONDecodeError:

                st.error(
                    "Invalid JSON response from backend."
                )

                st.text(response.text)

            except Exception as e:

                st.error(f"Unexpected Error: {e}")

else:

    st.info("Upload dataset to begin")
