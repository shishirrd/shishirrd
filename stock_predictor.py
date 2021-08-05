#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Code for hiding warnings
import warnings
warnings.filterwarnings("ignore")


# In[2]:


from statsmodels.tsa.holtwinters import ExponentialSmoothing


# In[3]:


import pandas as pd
import seaborn as sns
import numpy as np
import bsedata
import streamlit as st


# In[4]:


#Data Source
import yfinance as yf

#Data viz
import plotly.graph_objs as go


# In[5]:


from datetime import date


# In[ ]:


from dateutil.relativedelta import relativedelta
from datetime import date


# In[6]:


START = '2017-04-01'


# In[7]:


TODAY = date.today().strftime('%Y-%m-%d')


# In[ ]:


st.title("Tech stock prediction app")


# In[ ]:


stocks = ('HCLTECH.NS', 'AAPL', 'GOOG', 'AMD', 'NVDA', 'INTC',  
         'FB', 'RELIANCE.NS', 'MSFT', 'INFY.NS')


# In[ ]:


selected_stocks = st.selectbox("Select which stock you want to predict", stocks)


# In[ ]:


@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data


# In[ ]:


#data_load_state = st.text("Loading stock data...")
data = load_data(selected_stocks)
#data_load_state.text("Loading stock data...Done!")


# In[ ]:


st.write("Last traded price is: ") 
st.write(round(data['Close'].iloc[-1],2))


# In[ ]:


n_years = st.slider("Year to predict", 1, 4)


# In[ ]:


period = n_years * 365


# In[ ]:


def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))    
    fig.layout.update(title_text = 'Time Series Data', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)


# In[ ]:


plot_raw_data()


# In[ ]:


st.subheader("Here's what the past 5 days looked like!")
st.write(data.tail())


# In[ ]:


#ETS Forecasting
df_train = data.drop(['Open', 'High', 'Low', 'Adj Close', 'Volume'], axis='columns')
df_train = df_train.rename(columns={'Date':'DS', 'Close':'y'})


# In[ ]:


#Set index frequency as start of the day
df_train['DS'].freq = 'DS'


# In[ ]:


#Build an multiplicative ETS model
HWmodel2 = ExponentialSmoothing(df_train['y'], trend = 'mul', seasonal = 'mul', seasonal_periods=2).fit()


# In[ ]:


#Build regression model based on multiplicative ETS model
HWpred2 = HWmodel2.forecast(period)


# In[ ]:


#Wrap predictions into a pandas dataframe
prediction_data2 = pd.DataFrame(HWpred2, columns = ['Close'])


# In[ ]:


#Reset index
prediction_data2.reset_index(inplace=True)


# In[ ]:


#Add date column and wrap into a pandas dataframe
prediction_data2['Date'] = pd.DataFrame(pd.date_range(start=pd.to_datetime(date.today().strftime('%Y-%m-%d')), 
                                                     end=pd.to_datetime(date.today().strftime('%Y-%m-%d')) + 
                                                     np.timedelta64(period,'D'), freq='D'), columns=['Date'])


# In[ ]:


def plot_prediction2_data():
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=prediction_data2['Date'], y=prediction_data2['Close'], name='stock_forecast'))
    fig3.layout.update(title_text = 'ETS forecasted stock price', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig3)


# In[ ]:


plot_prediction2_data()


# In[ ]:


#Brief display of our forecast 
st.subheader("Here's what the last 5 days of our prediction looks like!")
st.write(prediction_data2['Close'].tail())


# #ETS Forecasting
# df_train = data.drop(['Open', 'High', 'Low', 'Adj Close', 'Volume'], axis='columns')
# df_train = df_train.rename(columns={'Date':'DS', 'Close':'y'})

# #Set index frequency as start of the day
# df_train['DS'].freq = 'DS'

# #Build an additive ETS model i.e. the forecasted value is the sum of baseline, trend and seasonality
# HWmodel1 = ExponentialSmoothing(df_train['y'], trend = 'add', seasonal = 'add', seasonal_periods=2).fit()

# #Build regression model based on additive ETS model
# HWpred1 = HWmodel1.forecast(period)

# #Wrap predictions into a pandas dataframe
# prediction_data = pd.DataFrame(HWpred1, columns = ['Close'])

# #Reset index
# prediction_data.reset_index(inplace=True)

# #Add date column and wrap into a pandas dataframe
# prediction_data['Date'] = pd.DataFrame(pd.date_range(start=pd.to_datetime(date.today().strftime('%Y-%m-%d')), 
#                                                      end=pd.to_datetime(date.today().strftime('%Y-%m-%d')) + 
#                                                      np.timedelta64(period,'D'), freq='D'), columns=['Date'])

# #Brief display of our forecast 
# st.subheader("Here's what the last 5 days of our prediction looks like!")
# st.write(prediction_data.tail())

# def plot_prediction_data():
#     fig2 = go.Figure()
#     fig2.add_trace(go.Scatter(x=prediction_data['Date'], y=prediction_data['Close'], name='stock_forecast'))
#     fig2.layout.update(title_text = 'ETS (add) forecasted stock price', xaxis_rangeslider_visible=True)
#     st.plotly_chart(fig2)

# plot_prediction_data()

# data = yf.download('INTC', START, TODAY, parse_dates=True)

# data.reset_index(inplace=True)

# #Forecasting
# df_train = data.drop(['Open', 'High', 'Low', 'Adj Close', 'Volume'], axis='columns')
# df_train = df_train.rename(columns={'Date':'DS', 'Close':'y'})

# #Set index frequency as start of the months
# df_train['DS'].freq = 'DS'

# #Build an additive ETS model i.e. the forecasted value is the sum of baseline, trend and seasonality
# HWmodel1 = ExponentialSmoothing(df_train['y'], trend = 'add', seasonal = 'add', seasonal_periods=90).fit()

# #Build regression model based on additive ETS model
# HWpred1 = HWmodel1.forecast(90)

# HWpred1

# prediction_data = pd.DataFrame(HWpred1, columns = ['Close'])
# prediction_data

# 

# prediction_data.reset_index(inplace=True)

# prediction_data['Date'] = df_train['DS'].iloc[0]+np.timedelta64(1,'D')
# prediction_data

# pd.to_datetime(date.today().strftime('%Y-%m-%d'))

# prediction_data['Date'] = pd.DataFrame(pd.date_range(start=pd.to_datetime(date.today().strftime('%Y-%m-%d')), 
#                                                      end=pd.to_datetime(date.today().strftime('%Y-%m-%d')) + np.timedelta64(90,'D'),
#                                                      freq='D'), columns=['Date'])

# prediction_data

# def plot_prediction_data():
#     fig2 = go.Figure()
#     fig2.add_trace(go.Scatter(x=prediction_data['Date'], y=prediction_data['Close'], name='stock_forecast'))
#     fig2.layout.update(title_text = 'Forecasted stock', xaxis_rangeslider_visible=True)
#     st.plotly_chart(fig2)

# plot_prediction_data()
