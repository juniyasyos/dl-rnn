"""
Text RNN model for word sequence prediction using LSTM.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


class TextRNN:
    """
    LSTM-based RNN for text/word sequence prediction.
    """
    
    def __init__(self, vocab_size, embedding_dim=100, lstm_units=128, 
                 sequence_length=20, dropout_rate=0.2):
        """
        Initialize TextRNN model.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            lstm_units: Number of LSTM units
            sequence_length: Length of input sequences
            dropout_rate: Dropout rate for regularization
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.sequence_length = sequence_length
        self.dropout_rate = dropout_rate
        self.model = None
        self.history = None
        
    def build_model(self):
        """Build the LSTM model architecture."""
        self.model = Sequential([
            # Word embedding layer
            Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_dim,
                input_length=self.sequence_length,
                name='embedding'
            ),
            
            # First LSTM layer
            LSTM(
                units=self.lstm_units,
                return_sequences=True,
                dropout=self.dropout_rate,
                recurrent_dropout=self.dropout_rate,
                name='lstm_1'
            ),
            
            # Second LSTM layer
            LSTM(
                units=self.lstm_units // 2,
                dropout=self.dropout_rate,
                recurrent_dropout=self.dropout_rate,
                name='lstm_2'
            ),
            
            # Dense layers
            Dense(units=self.lstm_units // 2, activation='relu', name='dense_1'),
            Dropout(self.dropout_rate),
            Dense(units=self.vocab_size, activation='softmax', name='output')
        ])
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              epochs=50, batch_size=32, verbose=1):
        """
        Train the model.
        
        Args:
            X_train: Training input sequences
            y_train: Training target sequences (one-hot encoded)
            X_val: Validation input sequences
            y_val: Validation target sequences
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
                patience=10,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
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
    
    def predict_next_word(self, sequence):
        """
        Predict the next word given a sequence.
        
        Args:
            sequence: Input sequence (as indices)
        
        Returns:
            Predicted word index and probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Prepare input
        input_seq = np.array(sequence).reshape(1, -1)
        
        # Make prediction
        prediction = self.model.predict(input_seq, verbose=0)
        predicted_word_idx = np.argmax(prediction[0])
        
        return predicted_word_idx, prediction[0]
    
    def generate_text(self, seed_sequence, num_words=20, temperature=1.0):
        """
        Generate text using the trained model.
        
        Args:
            seed_sequence: Starting sequence (as indices)
            num_words: Number of words to generate
            temperature: Sampling temperature (higher = more random)
        
        Returns:
            Generated sequence (as indices)
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        generated = seed_sequence.copy()
        
        for _ in range(num_words):
            # Use last sequence_length words
            input_seq = generated[-self.sequence_length:]
            input_seq = np.array(input_seq).reshape(1, -1)
            
            # Predict
            prediction = self.model.predict(input_seq, verbose=0)[0]
            
            # Apply temperature
            if temperature != 1.0:
                prediction = np.log(prediction + 1e-8) / temperature
                prediction = np.exp(prediction)
                prediction = prediction / np.sum(prediction)
            
            # Sample next word
            next_word = np.random.choice(len(prediction), p=prediction)
            generated.append(next_word)
        
        return generated
    
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


def create_text_model(vocab_size, embedding_dim=100, lstm_units=128, 
                     sequence_length=20, dropout_rate=0.2):
    """
    Factory function to create a TextRNN model.
    
    Args:
        vocab_size: Size of vocabulary
        embedding_dim: Dimension of word embeddings
        lstm_units: Number of LSTM units
        sequence_length: Length of input sequences
        dropout_rate: Dropout rate
    
    Returns:
        TextRNN instance
    """
    return TextRNN(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        lstm_units=lstm_units,
        sequence_length=sequence_length,
        dropout_rate=dropout_rate
    )


if __name__ == "__main__":
    # Example usage
    print("TextRNN model module loaded successfully!")
    
    # Create a sample model
    model = create_text_model(vocab_size=1000, sequence_length=20)
    model.build_model()
    print("\nModel architecture:")
    model.get_model_summary()