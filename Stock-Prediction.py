import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
from datetime import datetime
from keras.models import load_model

def predict_stock(Data):
    if isinstance(Data, pd.DataFrame):
        df=Data.reset_index()
    else:
        df=pd.read_csv(Data)
    
      
    dates=pd.to_datetime(df['Date'])
    
    scaler=MinMaxScaler(feature_range=(0,1))
    scaled_data=scaler.fit_transform(df['Close'].values.reshape(-1,1))
    sequence_length=30
    x=[]
    for i in range(sequence_length,len(df)):
        x.append(scaled_data[i-sequence_length:i,0])
    x=np.array(x)
    
    model= load_model('lstm_model.h5')
    future_prediction=model.predict(x)
    future_prediction=scaler.inverse_transform(future_prediction)
    last_date=dates.iloc[-1]
    forcasted_dates=pd.date_range(start=last_date+pd.DateOffset(1),periods=len(future_prediction),freq='B')
    
    prediction=pd.DataFrame({'Date':forcasted_dates,'Forecasted Price:':future_prediction.flatten()})
    return prediction



def plot_stock_forecast(Company_data, name):
    fig,ax= plt.subplots(figsize=(6,6))
    ax.plot( Company_data['Date'],Company_data['Forecasted Price:'], label=name)
    ax.set_xlabel('Date')
    ax.set_ylabel('Stock Price')
    ax.set_title('Forecasted Stock Prices')
    ax.legend()
    ax.tick_params(axis='x', rotation=45)
    plt.grid(True)
    st.pyplot(fig)

def downloader(company,ticker_df):
    data=pd.DataFrame()
    for index,r in ticker_df.iterrows():
        if (company==r['Name'] or company==r['Ticker']):
            ticker=r['Ticker']
            name=r['Name']
            break
    else:
      return None,None
    startDate='2019-01-03'
    endDate='2023-03-06'
    data=yf.download(ticker,start=startDate,end=endDate)
    return data,name

@st.cache_data
def preloaded_plot():
    Microsoft=predict_stock('datasets/MSFT (1).csv')
    Google=predict_stock('datasets/GOOGL.csv')
    Nvidia=predict_stock('datasets/NVDA.csv')
    Meta=predict_stock('datasets/META.csv')


    data = [
        (Microsoft['Date'], Microsoft['Forecasted Price:'], 'Microsoft'),
        (Google['Date'], Google['Forecasted Price:'], 'Google'),
        (Nvidia['Date'], Nvidia['Forecasted Price:'], 'Nvidia'),
        (Meta['Date'], Meta['Forecasted Price:'], 'Meta')
    ]

    num_plots = len(data)
    num_cols = 2
    num_rows = (num_plots + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))
    axes = axes.flatten()

    for i, (x, y, label) in enumerate(data):
        axes[i].plot(x, y, label=label)
        axes[i].set_xlabel('Date')
        axes[i].set_ylabel('Stock Price')
        axes[i].set_title(label)
        axes[i].legend()
        axes[i].tick_params(axis='x', rotation=45)

    # Hide unused subplots
    for j in range(num_plots, num_cols * num_rows):
        fig.delaxes(axes[j])

    # Adjust spacing between subplots
    plt.tight_layout()
    st.pyplot(fig)

def main():
    st.title("Stock price prediction")
    
    
    
    
    ticker_df=pd.read_excel('datasets/Yahoo Ticker Symbols - September 2017.xlsx')
    ticker_df=ticker_df[['Ticker','Name']]
    st.sidebar.title("Options")
    
    show_forecast=st.sidebar.checkbox('Show Forecasted Prices',value=True)
    show=st.sidebar.checkbox('Show Other Companies',value=True)
    company_name = st.sidebar.text_input("Enter Company Name or Ticker")
    if show:
      preloaded_plot()
    if show_forecast:
        data, name = downloader(company_name,ticker_df)
        if isinstance(data, pd.DataFrame):
            Company_data = predict_stock(data)
            plot_stock_forecast(Company_data, name)
            st.subheader('Forecasted Stock Prices')
            st.dataframe(Company_data)
        else:
            st.error("Company not found.")

    st.sidebar.markdown("---")
    st.sidebar.text("Created by: J-Abdullah")

if __name__=='__main__':
    main()












