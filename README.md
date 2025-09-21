# RNN Time Series Project

A structured project for time series prediction using Recurrent Neural Networks (RNN) with Keras/TensorFlow.

## Project Structure

```
rnn/
├── data/                   # Data files and datasets
├── notebooks/             # Jupyter notebooks for analysis
├── src/                   # Source code modules
├── models/               # Saved model files
├── README.md            # This file
└── requirements.txt     # Python dependencies
```

## Features

- **Text Time Series**: Word/character sequence prediction using LSTM/GRU
- **Stock Price Prediction**: Financial time series forecasting
- **Dummy Data Generation**: Synthetic datasets for experimentation
- **Modular Design**: Clean separation of data, models, and utilities

## Usage

1. Install dependencies: `pip install -r requirements.txt`
2. Run the main notebook: `notebooks/rnn_time_series_analysis.ipynb`
3. Explore individual models in the `src/` directory

## Models

- **Text RNN**: LSTM-based model for word sequence prediction
- **Stock RNN**: GRU-based model for stock price forecasting

## Dependencies

- TensorFlow/Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn