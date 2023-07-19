import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model
from keras.models import Sequential
from keras.layers import Dense, LSTM
import keras.backend as K
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.models import load_model
from datetime import datetime, timedelta
import joblib
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model

def get_stock_data(stock_symbol, start_date, end_date):
    data = yf.download(stock_symbol, start=start_date, end=end_date)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    data = data.reindex(date_range)
    data['Date'] = data.index
    data.fillna(method='ffill', inplace=True)
    stock_info = yf.Ticker(stock_symbol).info
    stock_name = stock_info.get('longName', 'N/A')
    return data,stock_name
def calculate_7d_average(data):
    return np.mean(data['Close'].tail(7))

def calculate_7d_high(data):
    return np.max(data['High'].tail(7))

def calculate_7d_low(data):
    return np.min(data['Low'].tail(7))

def train_lstm_model(data):
    output_var = pd.DataFrame(data['Adj Close'])
    features = ['Open']

    scaler = MinMaxScaler()
    feature_transform = scaler.fit_transform(data[features])
    feature_transform = pd.DataFrame(columns=features, data=feature_transform, index=data.index)

    timesplit = TimeSeriesSplit(n_splits=10)
    for train_index, test_index in timesplit.split(feature_transform):
        X_train, X_test = feature_transform[:len(train_index)], feature_transform[len(train_index): (len(train_index) + len(test_index))]
        y_train, y_test = output_var[:len(train_index)].values.ravel(), output_var[len(train_index): (len(train_index) + len(test_index))].values.ravel()

    trainX = np.array(X_train)
    testX = np.array(X_test)
    X_train = trainX.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test = testX.reshape(X_test.shape[0], 1, X_test.shape[1])

    lstm = Sequential()
    lstm.add(LSTM(32, input_shape=(1, trainX.shape[1]), return_sequences=False))
    lstm.add(Dense(1))
    lstm.compile(loss='mean_squared_error', optimizer='adam')

    history = lstm.fit(X_train, y_train, epochs=100, batch_size=8, verbose=1, shuffle=False)

    return lstm, scaler

def main():
    st.write("""
    # Stock Prediction Model using LSTM

    This app uses an LSTM (Long Short-Term Memory) model to predict stock prices based on historical data. 
    You can enter the stock symbol, start date, and end date to analyze the historical stock data and make predictions.
    """)    
    #Hiding the watermark
    hide_streamlit_style = """
    <style>
    footer {visibility: hidden;}
    </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    stock_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']  # Add more stock symbols as needed
    stock_symbol = st.selectbox('Select the stock symbol:', stock_symbols)    
    start_date = st.date_input('Select the start date:',datetime.now() - timedelta(days=3652))
    end_date = st.date_input('Select the end date:')

    if stock_symbol and start_date and end_date:
        if stock_symbol!='AAPL':
            st.subheader('Prediction Model is currently only supported for AAPL stock. Please select AAPL from the dropdown.')
        else:
            loaded_model = load_model(f'./models/{stock_symbol}_model.h5')
            scaler = joblib.load(f'./models/{stock_symbol}_scaler.joblib')

            stock_data = yf.Ticker(stock_symbol)
            latest_data = stock_data.history(period='1d')
            openp = latest_data['Open'][0]
            scaledopenp = scaler.fit_transform([[openp]])
            scaledopenp = scaledopenp.reshape(1, 1, scaledopenp.shape[1])
            expectedclosep = loaded_model.predict(scaledopenp)
            expectedclosep = scaler.inverse_transform(expectedclosep)

            st.subheader('Model Prediction')
            action_text = '<div style="text-align:center; margin-top: 1px;font-size:24px">'
            if expectedclosep[0, 0] > openp:
                action_text += f'''<strong>Latest Stock Price for {stock_symbol}:</strong> ${openp:.2f}<br>
                    <strong>Predicted Close Price for {stock_symbol}:</strong> ${expectedclosep[0, 0]:.2f}<br>
                <span style="color:green">Go Long (Buy)</span>'''
            else:
                action_text += f'''<strong>Latest Stock Price for {stock_symbol}:</strong> ${openp:.2f}<br>
                    <strong>Predicted Close Price for {stock_symbol}:</strong> ${expectedclosep[0, 0]:.2f}<br>
                <span style="color:red">Go Short (Sell)</span>'''
            action_text += '</div>'

            st.markdown(
                f'<div style="border: 2px solid #4CAF50; border-radius: 10px; padding: 15px;">{action_text}</div>',
                unsafe_allow_html=True
            )

        stock_data = yf.Ticker(stock_symbol)
        latest_data = stock_data.history(period='1d')
        data_24h = stock_data.history(period='1d')
        data_7d = stock_data.history(period='7d')

        current_price = latest_data['Close'].iloc[-1]
        high_24h = data_24h['High'].max()
        low_24h = data_24h['Low'].min()
        volume_24h = data_24h['Volume'].iloc[-1]
        average_7d = calculate_7d_average(data_7d)
        high_7d = calculate_7d_high(data_7d)
        low_7d = calculate_7d_low(data_7d)
        volume_7d = data_7d['Volume'].sum()
        
        st.subheader('Stock Details')
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                f'''
                <div style="center: 2px solid #4CAF50; border-radius: 10px; padding: 15px;">
                    <p><strong>24H High:</strong> ${high_24h:.2f}</p>
                    <p><strong>24H Low:</strong> ${low_24h:.2f}</p>
                    <p><strong>24H Volume:</strong> ${volume_24h}</p>
                </div>
                ''',
                unsafe_allow_html=True
            )

        with col2:
            st.markdown(
                f'''
                <div style="center: 2px solid #4CAF50; border-radius: 10px; padding: 15px;">
                    <p><strong>7D High:</strong> ${high_7d:.2f}</p>
                    <p><strong>7D Low:</strong> ${low_7d:.2f}</p>
                    <p><strong>7D Volume:</strong> ${volume_7d}</p>
                </div>
                ''',
                unsafe_allow_html=True
            )
        
        data,stock_name = get_stock_data(stock_symbol, start_date, end_date)
        st.subheader(f'Stock Data of {stock_name} (${stock_symbol.upper()})')
        st.write(data)
        st.subheader('Adjusted Close Price')
        st.line_chart(data['Adj Close'])

    st.markdown(
        """
        <div style="text-align:center; margin-top: 50px;">
            <span style="font-size: 18px;">Made with ❤️ by Sashank Ravipati</span> <br>
            <a href="https://www.linkedin.com/in/sashankravipati/" target="_blank"><img src="https://img.shields.io/badge/-LinkedIn-blue?style=flat-square&logo=Linkedin&logoColor=white" height="24"></a>    
            <a href="https://github.com/rsashank" target="_blank"><img src="https://img.shields.io/badge/-GitHub-black?style=flat-square&logo=GitHub&logoColor=white" height="24"></a>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == '__main__':
    main()