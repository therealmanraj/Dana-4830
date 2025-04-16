import streamlit as st
import boto3
import pandas as pd
import plotly.graph_objects as go

region = 'ca-central-1'
s3_bucket = 'dana-minicapstone-ca'
s3_incoming_key = 'incoming/hvac_test.csv'
s3_prediction_key = 'predictions/hvac_test.csv'
s3 = boto3.client('s3', region_name=region)

@st.cache_data(show_spinner=False)
def load_data():
    response_incoming = s3.get_object(Bucket=s3_bucket, Key=s3_incoming_key)
    response_prediction = s3.get_object(Bucket=s3_bucket, Key=s3_prediction_key)
    
    df = pd.read_csv(response_incoming['Body'])
    df_predictions = pd.read_csv(response_prediction['Body'])
    
    n_predictions = len(df_predictions)
    df = df.tail(n_predictions)
    df.reset_index(drop=True, inplace=True)
    
    df.columns = df.columns.str.replace(r"^b'|'$|\[.*?\]", "", regex=True)
    
    df = df.loc[df['Environment:Site Day Type Index'] != 0]
    
    df["HVAC_kWh"] = df["Electricity:HVAC"] * 2.77778e-7
    df.drop(columns='Electricity:HVAC', inplace=True)
    
    df_predictions['Predicted_HVAC_kWh'] = df_predictions['Predicted_HVAC_kWh'].astype(float)
    
    return df, df_predictions

df, df_predictions = load_data()

n_predictions = len(df_predictions)
avg_actual = df['HVAC_kWh'].mean()
avg_pred = df_predictions['Predicted_HVAC_kWh'].mean()

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df['HVAC_kWh'],
        mode='lines',
        name='Actual'
    )
)

fig.add_trace(
    go.Scatter(
        x=df_predictions.index,
        y=df_predictions['Predicted_HVAC_kWh'],
        mode='lines',
        name='Predicted',
        line=dict(color='yellow', dash='dash'),
        opacity=0.7
    )
)

fig.update_layout(
    title='HVAC Energy Consumption by Hour',
    xaxis_title='TimeFrame',
    yaxis_title='HVAC_kWh',
    hovermode='x unified'
)

st.title("HVAC Energy Consumption Dashboard")

st.plotly_chart(fig, use_container_width=True)

st.subheader("Key Metrics")

metrics_df = pd.DataFrame({
    "Number of Predictions": [n_predictions],
    "Avg Actual HVAC (kWh)": [f"{avg_actual:.6f}"],
    "Avg Predicted HVAC (kWh)": [f"{avg_pred:.6f}"]
})

html_table = metrics_df.to_html(index=False)
st.markdown(html_table, unsafe_allow_html=True)