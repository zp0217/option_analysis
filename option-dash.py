import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
from sklearn.metrics import r2_score, mean_squared_error

# Title
st.title("Option Volatility Analysis Dashboard")

# Load data with caching
def load_data():
    df = pd.read_csv("/data/gru_data.csv", parse_dates=["date"])
    return df

df = load_data()

# Sidebar filters
st.sidebar.header("Filters")
# Date range filter
dates = df["date"]
start_date, end_date = st.sidebar.date_input(
    "Select date range", 
    value=[dates.min(), dates.max()],
    min_value=dates.min(),
    max_value=dates.max()
)
# Strike price range filter
min_strike = int(df["strike_price"].min())
max_strike = int(df["strike_price"].max())
strike_range = st.sidebar.slider(
    "Strike Price Range", 
    min_value=min_strike, 
    max_value=max_strike, 
    value=(min_strike, max_strike),
    step=25
)

# Apply global filters (date and strike only)
df_filtered = df[
    (df.date >= pd.to_datetime(start_date)) &
    (df.date <= pd.to_datetime(end_date)) &
    (df.strike_price.between(strike_range[0], strike_range[1]))
]


# Time Series: Actual vs Predicted Volatility with Option Filter
st.subheader("Time Series: actual vs Predicted Volatility")
opt_choice = st.selectbox(
    "Select Option Type for Time Series", 
    ["All", "Put", "Call"]
)
flag_map = {"Put": "P", "Call": "C"}
if opt_choice in flag_map:
    df_ts = df_filtered[df_filtered.cp_flag == flag_map[opt_choice]]
else:
    df_ts = df_filtered

ds_ts = df_ts.groupby("date")[['impl_volatility', 'predicted_iv']].mean().reset_index()
fig_ts = px.line(
    ds_ts, 
    x="date", 
    y=["impl_volatility", "predicted_iv"], 
    labels={"value":"Volatility", "date":"Date", "variable":"Series"}
)
st.plotly_chart(fig_ts, use_container_width=True)

# Scatter: Predicted vs Actual for Put & Call separately
st.subheader("Scatter: Predicted vs Actual")
tabs = st.tabs(["Put Options", "Call Options"])
for tab, flag in zip(tabs, ["P", "C"]):
    with tab:
        df_flag = df_filtered[df_filtered["cp_flag"] == flag]
        if df_flag.empty:
            st.write(f"No data for {flag} options.")
        else:
            fig_scatter_flag = px.scatter(
                df_flag,
                x="impl_volatility",
                y="predicted_iv",
                labels={"impl_volatility": "Actual IV", "predicted_iv": "Predicted IV"}
            )
            st.plotly_chart(fig_scatter_flag, use_container_width=True)

# Time Series by Selected Strike Prices
st.subheader("Time Series by Strike Price")
available_strikes = sorted(df_filtered["strike_price"].unique())
selected_strikes = st.multiselect(
    "Select up to 5 strike prices", 
    available_strikes, 
    default=available_strikes[:5]
)
if selected_strikes:
    if len(selected_strikes) > 5:
        st.warning("Please select at most 5 strike prices.")
    else:
        df_strike = df_filtered[df_filtered["strike_price"].isin(selected_strikes)]
        df_long = df_strike.melt(
            id_vars=["date", "strike_price", "cp_flag"], 
            value_vars=["impl_volatility", "predicted_iv"], 
            var_name="Series", 
            value_name="Volatility"
        )
        tabs = st.tabs(["Put Options", "Call Options"])
        for tab, flag in zip(tabs, ["P", "C"]):
            with tab:
                df_flag = df_long[df_long["cp_flag"] == flag]
                if df_flag.empty:
                    st.write(f"No {flag} option data for selected strikes.")
                else:
                    fig_strike = px.line(
                        df_flag, 
                        x="date", 
                        y="Volatility", 
                        color="strike_price", 
                        facet_row="Series",
                        labels={"strike_price":"Strike", "Volatility":"IV", "date":"Date"}
                    )
                    st.plotly_chart(fig_strike, use_container_width=True)

