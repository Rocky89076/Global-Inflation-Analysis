import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title="PROBABILITY & STATISTICS PROJECT", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;600&display=swap');

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes pulseMetric {
    0% { text-shadow: 0 0 5px rgba(255, 0, 0, 0.5); }
    50% { text-shadow: 0 0 20px rgba(255, 0, 0, 1); }
    100% { text-shadow: 0 0 5px rgba(255, 0, 0, 0.5); }
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #000000;
    color: #FFFFFF;
}

.stApp {
    background-color: #000000;
    animation: fadeIn 0.8s ease-out;
}

h1, h2, h3, h4, h5, h6, .stMetric, .stSelectbox label {
    font-family: 'Space Mono', monospace;
}

.stSelectbox > div > div {
    background-color: #111111;
    color: #FFFFFF;
    border: 1px solid #333333;
    transition: all 0.3s ease;
}

.stSelectbox > div > div:hover {
    border-color: #FF0000;
    box-shadow: 0 0 10px rgba(255, 0, 0, 0.2);
}

.stExpander {
    border: none !important;
    background-color: #000000 !important;
    box-shadow: none !important;
}

[data-testid="stExpander"] details summary {
    font-family: 'Space Mono', monospace;
    color: #FFFFFF;
    border-bottom: 1px solid #333333;
    padding-bottom: 10px;
    transition: all 0.3s ease;
}

[data-testid="stExpander"] details summary:hover {
    color: #FF0000;
    padding-left: 8px;
    border-bottom: 1px solid #FF0000;
}

[data-testid="stMetricValue"] {
    color: #FF0000 !important;
    font-family: 'Space Mono', monospace;
    font-size: 3.5rem !important;
    animation: pulseMetric 3s infinite;
}

[data-testid="stMetricLabel"] {
    color: #FFFFFF !important;
    font-family: 'Space Mono', monospace;
    letter-spacing: 1px;
}

.stDataFrame {
    background-color: #000000;
}

div[data-testid="stDataFrame"] div[data-testid="StyledFullScreenButton"] {
    display: none;
}

[data-testid="stTable"] th {
    background-color: #111111 !important;
    color: #FFFFFF !important;
    font-family: 'Space Mono', monospace;
    border-bottom: 2px solid #FF0000 !important;
}

[data-testid="stTable"] td {
    color: #CCCCCC !important;
    border-bottom: 1px solid #333333 !important;
}
</style>
""", unsafe_allow_html=True)

st.title("GLOBAL INFLATION ANALYSIS")

@st.cache_data
def load_data():
    return pd.read_csv("global_inflation_post_covid.csv")

df = load_data()

country = st.selectbox("SELECT COUNTRY", df['country'].unique())

with st.spinner("FILTERING DATA..."):
    df_country = df[df['country'] == country].copy()
    if 'date' in df_country.columns:
        df_country['date'] = pd.to_datetime(df_country['date'])
        df_country = df_country.sort_values('date')

features = ['interest_rate', 'oil_price', 'gdp_growth', 'unemployment_rate', 'money_supply_m2', 'exchange_rate_usd', 'food_price_index', 'supply_chain_index']
target = 'inflation_rate'

with st.expander("TASK 2: SUMMARY STATISTICS"):
    numeric_df = df_country[features + [target]]
    desc = numeric_df.describe().T[['mean', '50%', '25%', '75%']]
    desc.columns = ['Mean', 'Median', 'Q1', 'Q3']
    desc['Mode'] = numeric_df.mode().iloc[0]
    desc = desc[['Mean', 'Median', 'Mode', 'Q1', 'Q3']]
    st.dataframe(desc, use_container_width=True)

with st.expander("TASK 3: BOX PLOTS"):
    df_melted = numeric_df.melt(var_name='Variable', value_name='Value')
    fig = px.box(df_melted, x='Variable', y='Value', template='plotly_dark', log_y=True)
    fig.update_traces(marker_color='#FF0000', line_color='#FFFFFF', fillcolor='#111111')
    fig.update_layout(
        plot_bgcolor='#000000',
        paper_bgcolor='#000000',
        xaxis_title=None,
        yaxis_title=None,
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor='#333333', zeroline=False)
    )
    st.plotly_chart(fig, use_container_width=True)

with st.expander("TASK 4: SCATTER PLOTS"):
    fig2 = px.scatter_matrix(numeric_df, dimensions=numeric_df.columns, template='plotly_dark')
    fig2.update_traces(marker=dict(color='#FF0000', opacity=0.5, size=4))
    fig2.update_layout(
        plot_bgcolor='#000000',
        paper_bgcolor='#000000',
        height=900,
        dragmode='zoom'
    )
    fig2.update_xaxes(showgrid=False, zeroline=False, tickfont=dict(color='#FFFFFF'))
    fig2.update_yaxes(showgrid=False, zeroline=False, tickfont=dict(color='#FFFFFF'))
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown("### INTERPRETATION")
    st.markdown("The scatter matrix visualizes pairwise relationships with interactive zooming. Hover over points to examine individual data coordinates. The red markers contrast sharply against the minimalist black background, revealing correlations and outliers critical for the inflation rate models.")

with st.expander("TASK 5: MODELING"):
    df_model = df_country.dropna(subset=features + [target])
    X = df_model[features]
    y = df_model[target]
    
    with st.spinner("CALCULATING MODEL VARIANCES..."):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        lr_pred = lr_model.predict(X_test)
        lr_mse = mean_squared_error(y_test, lr_pred)
        
        rf_model = RandomForestRegressor(random_state=42)
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        rf_mse = mean_squared_error(y_test, rf_pred)
        
        df_plot = df_country.dropna(subset=features + [target]).copy()
        if 'date' in df_plot.columns:
            df_plot = df_plot.sort_values('date')
        
        df_plot['LR_Pred'] = lr_model.predict(df_plot[features])
        df_plot['RF_Pred'] = rf_model.predict(df_plot[features])
        
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="LINEAR REGRESSION MSE", value=f"{lr_mse:.4f}")
    with col2:
        st.metric(label="RANDOM FOREST MSE", value=f"{rf_mse:.4f}")
        
    if 'date' in df_plot.columns:
        x_axis = 'date'
    else:
        df_plot['index'] = range(len(df_plot))
        x_axis = 'index'
        
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=df_plot[x_axis], y=df_plot[target], mode='lines', name='Actual', line=dict(color='#FFFFFF', width=2)))
    fig3.add_trace(go.Scatter(x=df_plot[x_axis], y=df_plot['LR_Pred'], mode='lines', name='LR Predicted', line=dict(color='#FF0000', width=1.5, dash='dash')))
    fig3.add_trace(go.Scatter(x=df_plot[x_axis], y=df_plot['RF_Pred'], mode='lines', name='RF Predicted', line=dict(color='#555555', width=1.5, dash='dot')))
    
    fig3.update_layout(
        template='plotly_dark',
        plot_bgcolor='#000000',
        paper_bgcolor='#000000',
        xaxis_title=None,
        yaxis_title=None,
        xaxis=dict(showgrid=True, gridcolor='#222222'),
        yaxis=dict(showgrid=True, gridcolor='#222222'),
        legend=dict(bgcolor='#000000', bordercolor='#333333', borderwidth=1),
        hovermode='x unified'
    )
    st.plotly_chart(fig3, use_container_width=True)
