import streamlit as st
import pandas as pd
import requests
import plotly.express as px

st.set_page_config(layout='wide')

st.title('🚀 AI Analytics Platform')

API_URL = "https://your-backend-name.onrender.com/analyze/"

uploaded_file = st.file_uploader(
    'Upload CSV or Excel',
    type=['csv', 'xlsx']
)

if uploaded_file:

    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)

    st.subheader('📊 Dataset Preview')

    st.dataframe(df.head())

    st.write('Shape:', df.shape)

    target = st.selectbox('Select Target Column', df.columns)

    chart = st.selectbox(
        'Visualization',
        ['Histogram', 'Scatter', 'Box', 'Line']
    )

    numeric_columns = df.select_dtypes(include='number').columns

    if len(numeric_columns) > 0:

        column = st.selectbox('Column', numeric_columns)

        if chart == 'Histogram':
            fig = px.histogram(df, x=column)

        elif chart == 'Scatter':
            fig = px.scatter(df, x=numeric_columns[0], y=numeric_columns[-1])

        elif chart == 'Box':
            fig = px.box(df, y=column)

        else:
            fig = px.line(df, y=column)

        st.plotly_chart(fig, use_container_width=True)

    if st.button('Run AI Analysis'):

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
            params={'target': target}
        )

        result = response.json()

        st.subheader('🤖 AutoML Results')

        st.json(result)

        scores = result['scores']

        score_df = pd.DataFrame({
            'Model': list(scores.keys()),
            'Score': list(scores.values())
        })

        st.dataframe(score_df)

        fig = px.bar(score_df, x='Model', y='Score')

        st.plotly_chart(fig, use_container_width=True)

        st.success(f"Best Model: {result['best_model']}")
