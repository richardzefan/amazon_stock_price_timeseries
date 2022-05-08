import streamlit as st
from datetime import date
import sys


import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image

import numpy as np
import warnings
warnings.filterwarnings("ignore")
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm




START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.markdown("![](https://github.com/richardzefan/amazon_stock_price_timeseries/blob/e3e18f074254610d9cd8cd7144e28499118873e5/TimeSeries.png)")
#st.image("https://github.com/richardzefan/amazon_stock_price_timeseries/blob/e3e18f074254610d9cd8cd7144e28499118873e5/TimeSeries.png", width=100)
st.title('Stock Forecast')

stocks = ['AMZN','GOOG', 'AAPL', 'MSFT', 'GME']
selected_stock = st.selectbox('Pilih Dataset', stocks)



@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

    
#data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
#data_load_state.text('Loading data... done!')

#tampilkan data
if selected_stock:
    st.subheader('Menampilkan Data '+ selected_stock)
number = st.number_input('Masukkan Jumlah Baris yang ingin ditampilkan', min_value=1,max_value=1000)
col1, col2 = st.columns(2)
with col1 :
    if st.button('Data Awal'):
        st.write(data.head(number))
        
        
with col2:
    if st.button('Data Terakhir'):
        st.write(data.tail(number))
def multi_selected_stock_chart():
    st.subheader('Plot data')
    newdf = data.drop('Date', axis='columns')
    multi_selected_stock = st.multiselect('Pilih kolom yang ingin diplotkan',list(newdf.columns))
    fig = go.Figure()
    for i in multi_selected_stock:
        fig.add_trace(go.Scatter(x=data['Date'], y=data[i],
                    mode='lines',
                    name=i))
    fig.layout.update(xaxis_rangeslider_visible=True)

    
    st.plotly_chart(fig)
multi_selected_stock_chart()

#===============================
#Cek missing value
st.subheader('Mengecek dan Mengisi Missing Value')
if st.button("Cek Missing Value"):
    st.write(data.isnull().sum())

st.caption("Abaikan button dibawah ini jika tidak ada nilai NaN")
if st.button('Mengisi Nilai NaN'):
        for i in data.columns[data.isnull().any(axis=0)]:     
            #---Applying Only on variables with NaN values
            data[i].fillna(data[i].mean(),inplace=True)
         
        st.write("""Hasil Setelah di ganti dengan nilai Mean
                 """)   
        st.write(data.isnull().sum())


#prophet function
def prophetMod(durasi):
              
    
            m = Prophet()
            m.fit(df_train)
            future = m.make_future_dataframe(periods=durasi)
            forecast = m.predict(future)
        
        
            # Show and plot forecast
            st.subheader('Forecast data')
            st.write(forecast.tail())
 
            st.write(f'Forecast untuk {n_years} tahun')
            fig1 = plot_plotly(m, forecast)
            st.plotly_chart(fig1)
        
            st.write("Forecast komponen")
            fig2 = m.plot_components(forecast)
            st.write(fig2)
#function ARIMA

def ArimaMod(durasi):
    # ARIMA
  
    model = ARIMA(df_close, order=(5,1,5))
    model_fit = model.fit(disp=-1)
    

    model_fit.plot_predict(1,len(df_close) + durasi) #Plot predictions for the next thousand days
    x = model_fit.forecast(steps=durasi) #Forecast the prediction for the next thousand days.
    x = plt.title(selected_stock+" Stock Forecast") #Add a stock title
    x = plt.xlabel("Year") #Add the year label to the bottom
    x = plt.ylabel(col1) #Add the open price to the y axis
    
    st.subheader('Hasil Forecasting')
    st.caption('Line Chart Hasil Forecasting')
    plt.show()
    fig = plt.gcf() # to get current figure
    ax = plt.gca() # to get current axes
    st.pyplot(fig)
    
    

    forecast=model_fit.forecast(steps=durasi)[0]
    #prediction=pd.DataFrame(forecast,index=df_close.index,columns=['Predicted Price'])
    df_close.index = pd.DatetimeIndex(df_close.index).to_period('D')
    da= df_close.sort_index()
    prediction=pd.DataFrame(forecast,columns=[f'Forecast {col1}'])
    st.line_chart(prediction)
    st.caption('Tabel Hasil Forecasting')
    st.table(prediction)


#====================================================
#tahun prediksi
#import SessionState
st.subheader('Melakukan Prediksi')

#====================================================

#forecast  

colw = ['Minggu','Bulan','Tahun']      
col0 = st.selectbox("Pilih",colw)
if (col0 == 'Minggu'):
    week = st.slider('Pilih Berapa Minggu untuk melakukan Prediksi:', 1,4)
    period1 =week*7
    
elif (col0 == 'Bulan'):
    month = st.slider('Pilih berapa Bulan untuk melakukan Prediksi (1 bulan = 30hari) :', 1,12)
    period2 = month*30 
    
elif (col0 == 'Tahun'):
    n_years = st.slider('Pilih Tahun untuk melakukan Prediksi:', 1, 10)
    period = n_years * 365    
  
pilModel = st.selectbox('Pilih Model',
('ARIMA','Prophet'))
col = ['Open','High','Low','Close','Volume']

col1 = st.selectbox("Pilih Kolom",col)


tombol = st.button('Prediksi')

if (col0 == 'Minggu'):
    if (pilModel =='Prophet'): 
        if col1 == 'Open':
            if tombol :
                # Predict forecast with Prophet.
                    df_train = data[['Date','Open']]
                    df_train = df_train.rename(columns={"Date": "ds", f"{'Open'}": "y"})
                    prophetMod(period1)
        elif col1 == 'High':
             if tombol :
                 # Predict forecast with Prophet.
                     df_train = data[['Date','High']]
                     df_train = df_train.rename(columns={"Date": "ds", f"{'High'}": "y"})
                     prophetMod(period1)           
        elif col1 == 'Low':
              if tombol :
                  # Predict forecast with Prophet.
                      df_train = data[['Date','Low']]
                      df_train = df_train.rename(columns={"Date": "ds", f"{'Low'}": "y"})
                      prophetMod(period1)     
        elif col1 == 'Close':
              if tombol :
                  # Predict forecast with Prophet.
                      df_train = data[['Date','Close']]
                      df_train = df_train.rename(columns={"Date": "ds", f"{'Close'}": "y"})
                      prophetMod(period1)
        elif col1 == 'Volume':
               if tombol :
                   # Predict forecast with Prophet.
                       df_train = data[['Date','Volume']]
                       df_train = df_train.rename(columns={"Date": "ds", f"{'Volume'}": "y"})
                       prophetMod(period1)
                       
    elif (pilModel =='ARIMA'):
                    
        if col1 == 'Open':
            if tombol :
           
                    df_close = data[["Date", "Open"]].copy() 
                    df_close["Date"] = pd.to_datetime(df_close["Date"]) 
                    df_close.set_index("Date", inplace = True) 
                    df_close = df_close.asfreq("b") 
                    df_close = df_close.fillna(method  = "bfill")
                    ArimaMod(period1)
                    
        elif col1 == 'High':
             if tombol :       
                    df_close = data[["Date", "High"]].copy() 
                    df_close["Date"] = pd.to_datetime(df_close["Date"]) 
                    df_close.set_index("Date", inplace = True) 
                    df_close = df_close.asfreq("b") 
                    df_close = df_close.fillna(method  = "bfill")
                    ArimaMod(period1)
        elif col1 == 'Low':
              if tombol :
                 
                  df_close = data[["Date", "Low"]].copy() 
                  df_close["Date"] = pd.to_datetime(df_close["Date"]) 
                  df_close.set_index("Date", inplace = True) 
                  df_close = df_close.asfreq("b") 
                  df_close = df_close.fillna(method  = "bfill")
                  ArimaMod(period1)
                     
        elif col1 == 'Close':
              if tombol :
                  
                  df_close = data[["Date", "Close"]].copy() 
                  df_close["Date"] = pd.to_datetime(df_close["Date"]) 
                  df_close.set_index("Date", inplace = True) 
                  df_close = df_close.asfreq("b") 
                  df_close = df_close.fillna(method  = "bfill")
                  ArimaMod(period1)
                      
        elif col1 == 'Volume':
               if tombol :
             
                   df_close = data[["Date", "Volume"]].copy() 
                   df_close["Date"] = pd.to_datetime(df_close["Date"]) 
                   df_close.set_index("Date", inplace = True) 
                   df_close = df_close.asfreq("b") 
                   df_close = df_close.fillna(method  = "bfill")
                   ArimaMod(period1)

elif (col0 == 'Bulan'):
    if (pilModel =='Prophet'): 
        if col1 == 'Open':
            if tombol :
                # Predict forecast with Prophet.
                    df_train = data[['Date','Open']]
                    df_train = df_train.rename(columns={"Date": "ds", f"{'Open'}": "y"})
                    prophetMod(period2)
        elif col1 == 'High':
             if tombol :
                 # Predict forecast with Prophet.
                     df_train = data[['Date','High']]
                     df_train = df_train.rename(columns={"Date": "ds", f"{'High'}": "y"})
                     prophetMod(period2)           
        elif col1 == 'Low':
              if tombol :
                  # Predict forecast with Prophet.
                      df_train = data[['Date','Low']]
                      df_train = df_train.rename(columns={"Date": "ds", f"{'Low'}": "y"})
                      prophetMod(period2)     
        elif col1 == 'Close':
              if tombol :
                  # Predict forecast with Prophet.
                      df_train = data[['Date','Close']]
                      df_train = df_train.rename(columns={"Date": "ds", f"{'Close'}": "y"})
                      prophetMod(period2)
        elif col1 == 'Volume':
               if tombol :
                   # Predict forecast with Prophet.
                       df_train = data[['Date','Volume']]
                       df_train = df_train.rename(columns={"Date": "ds", f"{'Volume'}": "y"})
                       prophetMod(period2)
                       
    elif (pilModel =='ARIMA'):
                    
        if col1 == 'Open':
            if tombol :
           
                    df_close = data[["Date", "Open"]].copy() 
                    df_close["Date"] = pd.to_datetime(df_close["Date"]) 
                    df_close.set_index("Date", inplace = True) 
                    df_close = df_close.asfreq("b") 
                    df_close = df_close.fillna(method  = "bfill")
                    ArimaMod(period2)
                    
        elif col1 == 'High':
             if tombol :       
                    df_close = data[["Date", "High"]].copy() 
                    df_close["Date"] = pd.to_datetime(df_close["Date"]) 
                    df_close.set_index("Date", inplace = True) 
                    df_close = df_close.asfreq("b") 
                    df_close = df_close.fillna(method  = "bfill")
                    ArimaMod(period2)
        elif col1 == 'Low':
              if tombol :
                 
                  df_close = data[["Date", "Low"]].copy() 
                  df_close["Date"] = pd.to_datetime(df_close["Date"]) 
                  df_close.set_index("Date", inplace = True) 
                  df_close = df_close.asfreq("b") 
                  df_close = df_close.fillna(method  = "bfill")
                  ArimaMod(period1)
                     
        elif col1 == 'Close':
              if tombol :
                  
                  df_close = data[["Date", "Close"]].copy() 
                  df_close["Date"] = pd.to_datetime(df_close["Date"]) 
                  df_close.set_index("Date", inplace = True) 
                  df_close = df_close.asfreq("b") 
                  df_close = df_close.fillna(method  = "bfill")
                  ArimaMod(period2)
                      
        elif col1 == 'Volume':
               if tombol :
             
                   df_close = data[["Date", "Volume"]].copy() 
                   df_close["Date"] = pd.to_datetime(df_close["Date"]) 
                   df_close.set_index("Date", inplace = True) 
                   df_close = df_close.asfreq("b") 
                   df_close = df_close.fillna(method  = "bfill")
                   ArimaMod(period2)                   

elif (col0 == 'Tahun'):
    if (pilModel =='Prophet'): 
        if col1 == 'Open':
            if tombol :
                # Predict forecast with Prophet.
                    df_train = data[['Date','Open']]
                    df_train = df_train.rename(columns={"Date": "ds", f"{'Open'}": "y"})
                    prophetMod(period)
        elif col1 == 'High':
             if tombol :
                 # Predict forecast with Prophet.
                     df_train = data[['Date','High']]
                     df_train = df_train.rename(columns={"Date": "ds", f"{'High'}": "y"})
                     prophetMod(period)           
        elif col1 == 'Low':
              if tombol :
                  # Predict forecast with Prophet.
                      df_train = data[['Date','Low']]
                      df_train = df_train.rename(columns={"Date": "ds", f"{'Low'}": "y"})
                      prophetMod(period)     
        elif col1 == 'Close':
              if tombol :
                  # Predict forecast with Prophet.
                      df_train = data[['Date','Close']]
                      df_train = df_train.rename(columns={"Date": "ds", f"{'Close'}": "y"})
                      prophetMod(period)
        elif col1 == 'Volume':
               if tombol :
                   # Predict forecast with Prophet.
                       df_train = data[['Date','Volume']]
                       df_train = df_train.rename(columns={"Date": "ds", f"{'Volume'}": "y"})
                       prophetMod(period)
                       
    elif (pilModel =='ARIMA'):
                    
        if col1 == 'Open':
            if tombol :
           
                    df_close = data[["Date", "Open"]].copy() 
                    df_close["Date"] = pd.to_datetime(df_close["Date"]) 
                    df_close.set_index("Date", inplace = True) 
                    df_close = df_close.asfreq("b") 
                    df_close = df_close.fillna(method  = "bfill")
                    ArimaMod(period)
                    
        elif col1 == 'High':
             if tombol :       
                    df_close = data[["Date", "High"]].copy() 
                    df_close["Date"] = pd.to_datetime(df_close["Date"]) 
                    df_close.set_index("Date", inplace = True) 
                    df_close = df_close.asfreq("b") 
                    df_close = df_close.fillna(method  = "bfill")
                    ArimaMod(period)
        elif col1 == 'Low':
              if tombol :
                 
                  df_close = data[["Date", "Low"]].copy() 
                  df_close["Date"] = pd.to_datetime(df_close["Date"]) 
                  df_close.set_index("Date", inplace = True) 
                  df_close = df_close.asfreq("b") 
                  df_close = df_close.fillna(method  = "bfill")
                  ArimaMod(period)
                     
        elif col1 == 'Close':
              if tombol :
                  
                  df_close = data[["Date", "Close"]].copy() 
                  df_close["Date"] = pd.to_datetime(df_close["Date"]) 
                  df_close.set_index("Date", inplace = True) 
                  df_close = df_close.asfreq("b") 
                  df_close = df_close.fillna(method  = "bfill")
                  ArimaMod(period)
                      
        elif col1 == 'Volume':
               if tombol :
             
                   df_close = data[["Date", "Volume"]].copy() 
                   df_close["Date"] = pd.to_datetime(df_close["Date"]) 
                   df_close.set_index("Date", inplace = True) 
                   df_close = df_close.asfreq("b") 
                   df_close = df_close.fillna(method  = "bfill")
                   ArimaMod(period)                   




    
