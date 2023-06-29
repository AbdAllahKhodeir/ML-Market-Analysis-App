# Import necessary libraries
import numpy as np
import pandas as pd
import pandas_datareader as pdr
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential, load_model
import yfinance as yfin
import streamlit as st
from PIL import Image
import mplfinance as mpf

# Load an image file
image = Image.open('market.jpeg')

# Set the Streamlit page title and favicon (emoji)
st.set_page_config(page_title='Tala Street Financial', page_icon=':money_with_wings:')

# Set the background color and font for the page
st.markdown('<style>body{background-color: #f5f5f5;font-family: Arial, sans-serif;}</style>', unsafe_allow_html=True)

# Display the header image on the page
st.image(image, caption='Market', use_column_width=True)

# Set the page title
st.title('Stock Trend Prediction')

# Set the sidebar with subheader, input fields, and date inputs
st.sidebar.subheader('Select Ticker and Time Period')
user_input = st.sidebar.text_input('Enter Stock Ticker', 'META')
start_date = st.sidebar.date_input('Start Date', value=pd.to_datetime('2000-01-01'))
end_date = st.sidebar.date_input('End Date', value=pd.to_datetime('2023-03-01'))

# Override the pandas_datareader to use yfinance as the data source
yfin.pdr_override()

# Download data using the user's input for ticker and date range
data = yfin.download(user_input, start_date, end_date)
# Remove the 'Adj Close' column from the data
data = data.drop(['Adj Close'], axis=1)
# Convert the index to a datetime object
data.index = pd.to_datetime(data.index)

# Display a subheader for the data description
st.subheader('Data from 2000 - 2023')
# Show a table with descriptive statistics for the data
st.write(data.describe())

# Visualize closing price
st.subheader('Closing Price vs Time Chart')
# Create a Matplotlib figure with specified dimensions
fig1, ax1 = plt.subplots(figsize=(12, 6))
# Plot the closing price data on the chart
ax1.plot(data.Close, 'b')
# Set x-axis label
ax1.set_xlabel('Time')
# Set y-axis label
ax1.set_ylabel('Price')
# Set the chart title
ax1.set_title('Closing Price vs Time Chart')
# Display the chart in the Streamlit app
st.pyplot(fig1)

# Visualize closing price with 100-day moving average
st.subheader('Closing Price vs Time Chart with 100 MA')
# Calculate the 100-day moving average
ma100 = data.Close.rolling(100).mean()
# Create a Matplotlib figure with specified dimensions
fig2, ax2 = plt.subplots(figsize=(12, 6))
# Plot the 100-day moving average on the chart
ax2.plot(ma100, 'r', label='MA100')
# Plot the closing price data on the chart
ax2.plot(data.Close, 'b', label=' Close Price')
# Set x-axis label
ax2.set_xlabel('Time')
# Set y-axis label
ax2.set_ylabel('Price')
# Set the chart title
ax2.set_title('Closing Price vs Time Chart with 100 MA')
# Display the legend for the chart
ax2.legend()
# Display the chart in the Streamlit app
st.pyplot(fig2)

# Visualize closing price with 100-day and 200-day moving averages
st.subheader('Closing Price vs Time Chart with 100 MA & 200 MA')
# Calculate the 200-day moving average
ma200 = data.Close.rolling(200).mean()
# Create a Matplotlib figure with specified dimensions
fig3, ax3 = plt.subplots(figsize=(12, 6))
# Plot the 100-day moving average on the chart
ax3.plot(ma100, 'r', label='MA100')
# Plot the 200-day moving average on the chart
ax3.plot(ma200, 'g', label='MA200')
# Plot the closing price data on the chart
ax3.plot(data.Close, 'b', label='Close Price')
# Set x-axis label
ax3.set_xlabel('Time')
# Set y-axis label
ax3.set_ylabel('Price')
# Set the chart title
ax3.set_title('Closing Price vs Time Chart with 100 MA & 200 MA')
# Display the legend for the chart
ax3.legend()
# Display the chart in the Streamlit app
st.pyplot(fig3)

# Create a candlestick chart for the last 100 days
st.subheader('Candlestick Chart - Last 100 Days')
candle_data = data.iloc[-100:][['Open', 'High', 'Low', 'Close', 'Volume']]
fig_candle, _ = mpf.plot(candle_data, type='candle', style='charles', title='Candlestick Chart - Last 100 Days',
                         ylabel='Price', volume=True, figsize=(12, 6), ylabel_lower='Volume', returnfig=True)
st.pyplot(fig_candle)

# Split the data into training and testing sets
data_training = pd.DataFrame(data['Close'][0:int(len(data)*0.70)])
data_testing = pd.DataFrame(data['Close'][int(len(data)*0.7): int(len(data))])

# Scale the training data
scaler = MinMaxScaler(feature_range=(0, 1))
data_training_array = scaler.fit_transform(data_training)

# Load the pre-trained model
model = load_model('model.h5')

# Prepare the test data
past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index=True)
input_data = scaler.fit_transform(final_df)

# Create test data arrays
x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
# Make predictions using the model
y_predicted = model.predict(x_test)
# Rescale the predictions and test data
scaler = scaler.scale_
scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor


# Final Graph
st.subheader('Prediction vs Original')
fig4 = plt.figure(figsize=(12, 6))
plt.plot(y_test, 'b', label='Actual Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig4)

# Create data_testing DataFrame with the same index as the last len(y_predicted) days of the original data
data_testing = data.iloc[-len(y_predicted):].copy()

# Create a DataFrame for predicted prices
predicted_prices = pd.DataFrame(y_predicted, index=data_testing.index, columns=['Predicted Close'])

# Combine original and predicted data
combined_data = pd.concat([data, predicted_prices], axis=1)

# Calculate Open, High, Low values for the predicted part
combined_data['Predicted Open'] = combined_data['Predicted Close'].shift(1)
combined_data['Predicted High'] = combined_data[['Predicted Open', 'Predicted Close']].max(axis=1)
combined_data['Predicted Low'] = combined_data[['Predicted Open', 'Predicted Close']].min(axis=1)

# Select the last 914 days (100 days actual + 814 days predicted)
candle_data_actual_predicted = combined_data.iloc[-(100 + len(y_predicted)):].copy()

# Replace original OHLC values with predicted values in the predicted part
candle_data_actual_predicted.loc[predicted_prices.index, ['Open', 'High', 'Low']] = candle_data_actual_predicted.loc[predicted_prices.index, ['Predicted Open', 'Predicted High', 'Predicted Low']].values

# Fill missing values in the combined dataset
candle_data_actual_predicted.fillna(method='ffill', inplace=True)

# Create a candlestick chart for actual past 100 days and predicted 100 days
st.subheader('Candlestick Chart - Actual Past 100 Days and Predicted 100 Days')
fig_candle_actual_predicted, _ = mpf.plot(candle_data_actual_predicted[['Open', 'High', 'Low', 'Close']], type='candle', style='charles', title='Candlestick Chart - Actual Past 100 Days and Predicted 100 Days',
                                          ylabel='Price', figsize=(12, 6), returnfig=True)
st.pyplot(fig_candle_actual_predicted)

# Add footer
st.markdown('---')
st.markdown('Created by AbdAllah Khodeir for Tala Street Financial.')

# Hide Streamlit menu
hide_streamlit_style = """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
