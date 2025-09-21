"""
Data generation utilities for creating dummy time series datasets.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random


def generate_text_data(n_samples=1000, vocab_size=50, sequence_length=20):
    """
    Generate dummy text data for word-level time series prediction.
    
    Args:
        n_samples: Number of sequences to generate
        vocab_size: Size of vocabulary (number of unique words)
        sequence_length: Length of each sequence
    
    Returns:
        sequences: List of word sequences
        word_to_idx: Word to index mapping
        idx_to_word: Index to word mapping
    """
    # Create dummy vocabulary
    words = [f"word_{i}" for i in range(vocab_size)]
    word_to_idx = {word: i for i, word in enumerate(words)}
    idx_to_word = {i: word for i, word in enumerate(words)}
    
    # Generate sequences with some patterns
    sequences = []
    for _ in range(n_samples):
        sequence = []
        for i in range(sequence_length):
            if i == 0:
                # Random start
                word_idx = random.randint(0, vocab_size - 1)
            else:
                # Add some pattern dependency
                prev_idx = sequence[i-1]
                if random.random() < 0.3:  # 30% chance of pattern
                    word_idx = (prev_idx + 1) % vocab_size
                else:
                    word_idx = random.randint(0, vocab_size - 1)
            sequence.append(word_idx)
        sequences.append(sequence)
    
    return sequences, word_to_idx, idx_to_word


def generate_stock_data(n_days=1000, start_price=100.0, volatility=0.02):
    """
    Generate dummy stock price data with realistic patterns.
    
    Args:
        n_days: Number of trading days
        start_price: Starting stock price
        volatility: Daily volatility (standard deviation of returns)
    
    Returns:
        DataFrame with Date, Open, High, Low, Close, Volume columns
    """
    np.random.seed(42)  # For reproducible results
    
    # Generate dates
    start_date = datetime(2020, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_days)]
    
    # Generate price data using geometric Brownian motion
    returns = np.random.normal(0.0005, volatility, n_days)  # Small positive drift
    prices = [start_price]
    
    for i in range(1, n_days):
        price = prices[-1] * (1 + returns[i])
        prices.append(max(price, 1.0))  # Prevent negative prices
    
    # Generate OHLC data
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        # Generate open price (close to previous close)
        if i == 0:
            open_price = close
        else:
            open_price = prices[i-1] * (1 + np.random.normal(0, volatility/4))
        
        # Generate high and low
        daily_range = abs(np.random.normal(0, volatility/2))
        high = max(open_price, close) * (1 + daily_range)
        low = min(open_price, close) * (1 - daily_range)
        
        # Generate volume
        volume = int(np.random.normal(1000000, 200000))
        volume = max(volume, 100000)  # Minimum volume
        
        data.append({
            'Date': date,
            'Open': round(open_price, 2),
            'High': round(high, 2),
            'Low': round(low, 2),
            'Close': round(close, 2),
            'Volume': volume
        })
    
    return pd.DataFrame(data)


def create_time_series_features(df, target_col='Close', window_sizes=[5, 10, 20]):
    """
    Create additional time series features for stock data.
    
    Args:
        df: DataFrame with stock data
        target_col: Target column name
        window_sizes: List of window sizes for moving averages
    
    Returns:
        Enhanced DataFrame with additional features
    """
    df = df.copy()
    
    # Moving averages
    for window in window_sizes:
        df[f'MA_{window}'] = df[target_col].rolling(window=window).mean()
    
    # Price changes
    df['Price_Change'] = df[target_col].pct_change()
    df['Price_Change_Abs'] = df['Price_Change'].abs()
    
    # Volatility (rolling standard deviation)
    df['Volatility_10'] = df['Price_Change'].rolling(window=10).std()
    
    # High-Low range
    df['HL_Range'] = (df['High'] - df['Low']) / df['Close']
    
    # Volume features
    df['Volume_MA_10'] = df['Volume'].rolling(window=10).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_10']
    
    return df


if __name__ == "__main__":
    # Generate and save text data
    text_sequences, word_to_idx, idx_to_word = generate_text_data()
    
    # Save text data
    np.save('/home/juni/Praktikum/deep-learning/rnn/data/text_sequences.npy', text_sequences)
    np.save('/home/juni/Praktikum/deep-learning/rnn/data/word_to_idx.npy', word_to_idx)
    np.save('/home/juni/Praktikum/deep-learning/rnn/data/idx_to_word.npy', idx_to_word)
    
    # Generate and save stock data
    stock_df = generate_stock_data()
    stock_df_enhanced = create_time_series_features(stock_df)
    
    stock_df.to_csv('/home/juni/Praktikum/deep-learning/rnn/data/stock_data.csv', index=False)
    stock_df_enhanced.to_csv('/home/juni/Praktikum/deep-learning/rnn/data/stock_data_features.csv', index=False)
    
    print("Dummy datasets generated successfully!")
    print(f"Text sequences: {len(text_sequences)} samples")
    print(f"Stock data: {len(stock_df)} days")