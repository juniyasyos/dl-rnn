"""
Stock RNN model for time series forecasting using GRU.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler


class StockRNN:
    """
    GRU-based RNN for stock price time series forecasting.
    """
    
    def __init__(self, n_features, sequence_length=60, gru_units=64, 
                 dropout_rate=0.2, dense_units=32):
        """
        Initialize StockRNN model.
        
        Args:
            n_features: Number of input features
            sequence_length: Length of input sequences (lookback period)
            gru_units: Number of GRU units
            dropout_rate: Dropout rate for regularization
            dense_units: Number of units in dense layers
        """
        self.n_features = n_features
        self.sequence_length = sequence_length
        self.gru_units = gru_units
        self.dropout_rate = dropout_rate
        self.dense_units = dense_units
        self.model = None
        self.history = None
        
    def build_model(self):
        """Build the GRU model architecture."""
        self.model = Sequential([
            # First GRU layer
            GRU(
                units=self.gru_units,
                return_sequences=True,
                input_shape=(self.sequence_length, self.n_features),
                dropout=self.dropout_rate,
                recurrent_dropout=self.dropout_rate,
                name='gru_1'
            ),
            BatchNormalization(),
            
            # Second GRU layer
            GRU(
                units=self.gru_units // 2,
                return_sequences=True,
                dropout=self.dropout_rate,
                recurrent_dropout=self.dropout_rate,
                name='gru_2'
            ),
            BatchNormalization(),
            
            # Third GRU layer
            GRU(
                units=self.gru_units // 4,
                dropout=self.dropout_rate,
                recurrent_dropout=self.dropout_rate,
                name='gru_3'
            ),
            
            # Dense layers
            Dense(units=self.dense_units, activation='relu', name='dense_1'),
            Dropout(self.dropout_rate),
            BatchNormalization(),
            
            Dense(units=self.dense_units // 2, activation='relu', name='dense_2'),
            Dropout(self.dropout_rate),
            
            # Output layer
            Dense(units=1, activation='linear', name='output')
        ])
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return self.model
    
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              epochs=100, batch_size=32, verbose=1):
        """
        Train the model.
        
        Args:
            X_train: Training input sequences
            y_train: Training target values
            X_val: Validation input sequences
            y_val: Validation target values
            epochs: Number of training epochs
            batch_size: Batch size
            verbose: Verbosity level
        
        Returns:
            Training history
        """
        if self.model is None:
            self.build_model()
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.3,
                patience=8,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Prepare validation data
        validation_data = (X_val, y_val) if X_val is not None else None
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return self.history
    
    def predict(self, X, verbose=0):
        """
        Make predictions.
        
        Args:
            X: Input sequences
            verbose: Verbosity level
        
        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        return self.model.predict(X, verbose=verbose)
    
    def predict_next_price(self, sequence):
        """
        Predict next price given a sequence.
        
        Args:
            sequence: Input sequence with shape (sequence_length, n_features)
        
        Returns:
            Predicted price
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Reshape input
        input_seq = sequence.reshape(1, self.sequence_length, self.n_features)
        
        # Make prediction
        prediction = self.model.predict(input_seq, verbose=0)
        
        return prediction[0][0]
    
    def predict_future(self, last_sequence, n_days=5):
        """
        Predict multiple future prices.
        
        Args:
            last_sequence: Last known sequence
            n_days: Number of days to predict
        
        Returns:
            Array of predicted prices
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(n_days):
            # Predict next value
            next_pred = self.predict_next_price(current_sequence)
            predictions.append(next_pred)
            
            # Update sequence (assuming we only predict the first feature - price)
            # Create new row with predicted price and estimated other features
            new_row = current_sequence[-1].copy()
            new_row[0] = next_pred  # Update price (assuming it's first feature)
            
            # Shift sequence and add new prediction
            current_sequence = np.vstack([current_sequence[1:], new_row])
        
        return np.array(predictions)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance.
        
        Args:
            X_test: Test input sequences
            y_test: Test target values
        
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Get predictions
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        mse = np.mean((y_test - y_pred.flatten()) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_test - y_pred.flatten()))
        
        # Calculate percentage errors
        mape = np.mean(np.abs((y_test - y_pred.flatten()) / y_test)) * 100
        
        # Direction accuracy (up/down prediction)
        y_test_diff = np.diff(y_test)
        y_pred_diff = np.diff(y_pred.flatten())
        direction_acc = np.mean(np.sign(y_test_diff) == np.sign(y_pred_diff)) * 100
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'Direction_Accuracy': direction_acc
        }
    
    def save_model(self, filepath):
        """Save the trained model."""
        if self.model is not None:
            self.model.save(filepath)
            print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model."""
        self.model = tf.keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
    
    def get_model_summary(self):
        """Get model architecture summary."""
        if self.model is not None:
            return self.model.summary()
        else:
            print("Model not built yet!")


class StockPredictor:
    """
    Complete stock prediction pipeline with preprocessing.
    """
    
    def __init__(self, sequence_length=60):
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler()
        self.model = None
        self.feature_columns = None
        
    def prepare_data(self, df, target_col='Close', feature_cols=None):
        """
        Prepare data for training.
        
        Args:
            df: DataFrame with stock data
            target_col: Target column name
            feature_cols: List of feature columns (if None, use all numeric)
        
        Returns:
            X_train, X_test, y_train, y_test
        """
        # Select features
        if feature_cols is None:
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        self.feature_columns = feature_cols
        
        # Remove rows with NaN
        df_clean = df[feature_cols].dropna()
        
        # Scale data
        scaled_data = self.scaler.fit_transform(df_clean)
        
        # Find target column index
        target_idx = feature_cols.index(target_col)
        
        # Create sequences
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.sequence_length:i])
            y.append(scaled_data[i, target_idx])
        
        X, y = np.array(X), np.array(y)
        
        # Split data
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        return X_train, X_test, y_train, y_test
    
    def build_and_train(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """Build and train the model."""
        # Create model
        self.model = StockRNN(
            n_features=X_train.shape[2],
            sequence_length=X_train.shape[1]
        )
        
        # Train model
        history = self.model.train(X_train, y_train, X_val, y_val, **kwargs)
        
        return history


def create_stock_model(n_features, sequence_length=60, gru_units=64, 
                      dropout_rate=0.2, dense_units=32):
    """
    Factory function to create a StockRNN model.
    
    Args:
        n_features: Number of input features
        sequence_length: Length of input sequences
        gru_units: Number of GRU units
        dropout_rate: Dropout rate
        dense_units: Number of dense units
    
    Returns:
        StockRNN instance
    """
    return StockRNN(
        n_features=n_features,
        sequence_length=sequence_length,
        gru_units=gru_units,
        dropout_rate=dropout_rate,
        dense_units=dense_units
    )


if __name__ == "__main__":
    # Example usage
    print("StockRNN model module loaded successfully!")
    
    # Create a sample model
    model = create_stock_model(n_features=5, sequence_length=60)
    model.build_model()
    print("\nModel architecture:")
    model.get_model_summary()