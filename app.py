import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# ✅ Title of the Streamlit App
st.title("📈 LSTM-Based Inventory Forecasting")

# ✅ Upload CSV File
uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])

if uploaded_file:
    # Load Data
    data = pd.read_csv(uploaded_file)
    df = data.copy()

    # Ensure 'Item' column is set as index
    if 'Item' in df.columns:
        df.set_index('Item', inplace=True)

    # ✅ Show Uploaded Data in a Table
    st.subheader("📋 Uploaded Data Preview")
    st.dataframe(df.head(10))  # Show first 10 rows

    # Convert only valid month columns to datetime format
    df.columns = [pd.to_datetime(col, format='%Y-%m') if '-' in str(col) else col for col in df.columns]

    # Convert datetime columns back to string format (YYYY-MM) for clean output
    df.columns = [col.strftime('%Y-%m') if isinstance(col, pd.Timestamp) else col for col in df.columns]

    # Compute EMA (Exponential Moving Average)
    df['EMA_3M'] = df.ewm(span=3, adjust=False, axis=1).mean().iloc[:, -1]
    df['EMA_6M'] = df.ewm(span=6, adjust=False, axis=1).mean().iloc[:, -1]

    # Extract Features & Time-Series Data
    features = df[['EMA_3M', 'EMA_6M']]
    time_series_data = df.drop(columns=['EMA_3M', 'EMA_6M'])

    # ✅ Define Lookback Period
    n_steps = 6  

    # ✅ Generate Future Month Names
    last_date = pd.to_datetime(time_series_data.columns[-1], format='%Y-%m')
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=3, freq='M')
    future_dates = [date.strftime('%Y-%m') for date in future_dates]

    # ✅ Forecast Dictionary
    forecast_dict = {item: time_series_data.loc[item].tolist() for item in time_series_data.index}

    # ✅ Streamlit Progress Bar
    progress_bar = st.progress(0)
    
    # ✅ Loop Through Each Item & Train LSTM Model
    for idx, item in enumerate(time_series_data.index):
        st.write(f"Training LSTM for Item: {item}...")

        # Get Item-Specific Data
        item_data = time_series_data.loc[item].values.reshape(-1, 1)

        # Scale Data
        scaler = MinMaxScaler()
        item_data_scaled = scaler.fit_transform(item_data)

        # Scale Features
        item_features = np.tile(features.loc[item].values, (len(item_data_scaled), 1))
        feature_scaler = MinMaxScaler()
        item_features_scaled = feature_scaler.fit_transform(item_features)

        # Concatenate Time-Series & Features
        combined_data = np.hstack((item_data_scaled, item_features_scaled))

        # Create Sequences
        def create_sequences(data, n_steps):
            X, y = [], []
            for i in range(len(data) - n_steps):
                X.append(data[i:i + n_steps])
                y.append(data[i + n_steps, 0])  
            return np.array(X), np.array(y)

        X, y = create_sequences(combined_data, n_steps)
        X = X.reshape((X.shape[0], X.shape[1], combined_data.shape[1]))

        # ✅ Build & Train LSTM Model
        model = Sequential([
            LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps, X.shape[2])),
            LSTM(50, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X, y, epochs=200, verbose=0)

        # ✅ Predict Next 3 Months
        future_predictions = []
        last_sequence = combined_data[-n_steps:]

        for _ in range(3):
            last_sequence = last_sequence.reshape((1, n_steps, combined_data.shape[1]))
            pred = model.predict(last_sequence, verbose=0)
            future_predictions.append(pred[0][0])

            # Shift Window
            new_entry = np.hstack([pred[0][0], features.loc[item].values])
            last_sequence = np.vstack([last_sequence[0][1:], new_entry])

        # Convert Predictions Back to Original Scale
        predictions_rescaled = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

        # Store Predictions
        forecast_dict[item].extend(predictions_rescaled.flatten())

        # Update Progress Bar
        progress_bar.progress((idx + 1) / len(time_series_data.index))

    # Convert Forecast Dictionary to DataFrame
    all_months = list(time_series_data.columns) + future_dates
    forecast_df = pd.DataFrame.from_dict(forecast_dict, orient='index', columns=all_months)

    # Add Features Back
    forecast_df = features.merge(forecast_df, left_index=True, right_index=True)

    # Reset Index
    forecast_df.reset_index(inplace=True)
    forecast_df.rename(columns={'index': 'Item'}, inplace=True)

    # ✅ Display Forecast Table
    st.subheader("📊 Forecasted Data")
    st.dataframe(forecast_df)  # Interactive table view

    # ✅ Download Button
    st.download_button("📥 Download Predictions as CSV", forecast_df.to_csv(index=False), file_name="future_forecast.csv")

