import streamlit as st
from datetime import date

from pandas import pandas as pd 

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly

from plotly import graph_objs as go





START='2015-01-01'

TODAY='2022-10-05'



st.title(" Stock Prediction and Analyser")

stocks=("AAPL","GOOG","MSFT","GME","CCC","TWTR","HSI","TOPS","BNGO","BLKB","SHOP","PEGY","SNTI","VEV","TESCO","BABA","1810.HK","IMPP","PUMSY","EVX","QLD","REMX","KARS","BLOK","GXG","HAIL","IEO","FXO","TOK","CS","COIN","CCL","TXG","WE","AFRM")
selected_stocks=st.selectbox("Select data for the prediction",stocks)

n_years=st.slider("Year of Prediction: ", 1 , 6)

period=n_years*365

@st.cache
def helper(x):
    l  = str(x).split()
    return l[0]
#using so that i do not have to download the data again and again 
def load_data(ticker):
    data=yf.download(ticker,START,TODAY)
    data.reset_index(inplace=True)# putting the dates in the first column
    data =pd.DataFrame(data)
    data["Date"] = data["Date"].apply(helper)

    return data


data_load_state=st.text("Load data...")
data=load_data(selected_stocks)
data_load_state.text("Loading data...done")

st.subheader('Raw data')
st.write(data.tail())

def plot_raw_data():
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Open'],name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Close'],name='stock_close'))
    
    fig.layout.update(title_text="Time Series Data",xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

# Forecasting 

df_train=data[['Date','Close']]
df_train=df_train.rename(columns={"Date": "ds", "Close": "y"})



m=Prophet()
m.fit(df_train)
future=m.make_future_dataframe(periods=period)
forecast=m.predict(future)
forecast=pd.DataFrame(forecast)
forecast['ds']=forecast['ds'].apply(helper)

st.subheader('Forecast data')
st.write(forecast.tail())


st.write('forecast data')
fig1=plot_plotly(m,forecast)
st.plotly_chart(fig1)



# import datetime
# def helper2(x):
#     return datetime.strptime(x, '%Y-%m-%d')
# forecast['ds'] = forecast['ds'].apply(helper2)
# st.write('Forecast Components')
# fig2=m.plot_components(forecast)
# st.write(fig2)