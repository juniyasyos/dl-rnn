"""
Utility functions for data preprocessing, model evaluation, and visualization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf


def prepare_text_sequences(sequences, sequence_length, vocab_size):
    """
    Prepare text sequences for RNN training.
    
    Args:
        sequences: List of word sequences (as indices)
        sequence_length: Length of input sequences
        vocab_size: Size of vocabulary
    
    Returns:
        X: Input sequences
        y: Target sequences (one-hot encoded)
    """
    X, y = [], []
    
    for sequence in sequences:
        if len(sequence) >= sequence_length + 1:
            for i in range(len(sequence) - sequence_length):
                # Input: sequence of length sequence_length
                X.append(sequence[i:i + sequence_length])
                # Target: next word (one-hot encoded)
                target = sequence[i + sequence_length]
                y_one_hot = np.zeros(vocab_size)
                y_one_hot[target] = 1
                y.append(y_one_hot)
    
    return np.array(X), np.array(y)


def prepare_stock_sequences(data, sequence_length=60, target_col='Close'):
    """
    Prepare stock data for RNN training.
    
    Args:
        data: DataFrame with stock data
        sequence_length: Number of previous days to use as input
        target_col: Column to predict
    
    Returns:
        X_train, X_test, y_train, y_test, scaler
    """
    # Use only numeric columns
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    df = data[numeric_cols].copy()
    
    # Remove rows with NaN values
    df = df.dropna()
    
    # Scale the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    
    # Find target column index
    target_idx = list(df.columns).index(target_col)
    
    # Create sequences
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i, target_idx])
    
    X, y = np.array(X), np.array(y)
    
    # Split into train and test
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    return X_train, X_test, y_train, y_test, scaler


def plot_training_history(history):
    """
    Plot training and validation loss/accuracy.
    
    Args:
        history: Keras training history object
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot loss
    axes[0].plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        axes[0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0].set_title('Model Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot accuracy (if available)
    if 'accuracy' in history.history:
        axes[1].plot(history.history['accuracy'], label='Training Accuracy')
        if 'val_accuracy' in history.history:
            axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
        axes[1].set_title('Model Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True)
    else:
        # Remove second subplot if no accuracy
        fig.delaxes(axes[1])
    
    plt.tight_layout()
    plt.show()


def plot_stock_predictions(y_true, y_pred, title="Stock Price Predictions"):
    """
    Plot actual vs predicted stock prices.
    
    Args:
        y_true: Actual prices
        y_pred: Predicted prices
        title: Plot title
    """
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='Actual', alpha=0.7)
    plt.plot(y_pred, label='Predicted', alpha=0.7)
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()


def calculate_metrics(y_true, y_pred):
    """
    Calculate regression metrics.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
    
    Returns:
        Dictionary with metrics
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    
    # Calculate percentage error
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape
    }


def plot_text_generation_sample(model, seed_sequence, idx_to_word, word_to_idx, 
                               vocab_size, num_words=20):
    """
    Generate and display text using trained model.
    
    Args:
        model: Trained text generation model
        seed_sequence: Starting sequence (as indices)
        idx_to_word: Index to word mapping
        word_to_idx: Word to index mapping
        vocab_size: Size of vocabulary
        num_words: Number of words to generate
    """
    generated_sequence = seed_sequence.copy()
    
    for _ in range(num_words):
        # Prepare input
        input_seq = np.array(generated_sequence[-len(seed_sequence):]).reshape(1, -1)
        
        # Predict next word
        prediction = model.predict(input_seq, verbose=0)
        next_word_idx = np.argmax(prediction[0])
        
        generated_sequence.append(next_word_idx)
    
    # Convert to words
    words = [idx_to_word[idx] for idx in generated_sequence]
    
    print("Generated text:")
    print(" ".join(words))
    
    return generated_sequence


def plot_feature_importance(data, target_col='Close'):
    """
    Plot correlation heatmap for stock features.
    
    Args:
        data: DataFrame with features
        target_col: Target column
    """
    # Calculate correlation matrix
    corr_matrix = data.corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, fmt='.2f')
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.show()
    
    # Show correlations with target
    target_corr = corr_matrix[target_col].abs().sort_values(ascending=False)
    print(f"\nCorrelations with {target_col}:")
    for feature, corr in target_corr.items():
        if feature != target_col:
            print(f"{feature}: {corr:.3f}")


def save_model_summary(model, filepath):
    """
    Save model architecture summary to file.
    
    Args:
        model: Keras model
        filepath: Path to save summary
    """
    with open(filepath, 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    print(f"Model summary saved to {filepath}")


class EarlyStopping:
    """Simple early stopping implementation."""
    
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience