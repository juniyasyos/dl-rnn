"""
Package initialization for RNN time series models.
"""

from .data_generator import (
    generate_text_data,
    generate_stock_data,
    create_time_series_features
)

from .text_rnn import TextRNN, create_text_model
from .stock_rnn import StockRNN, StockPredictor, create_stock_model
from .utils import (
    prepare_text_sequences,
    prepare_stock_sequences,
    plot_training_history,
    plot_stock_predictions,
    calculate_metrics,
    plot_text_generation_sample,
    plot_feature_importance
)

__version__ = "1.0.0"
__author__ = "RNN Time Series Project"

__all__ = [
    # Data generation
    'generate_text_data',
    'generate_stock_data',
    'create_time_series_features',
    
    # Models
    'TextRNN',
    'StockRNN',
    'StockPredictor',
    'create_text_model',
    'create_stock_model',
    
    # Utilities
    'prepare_text_sequences',
    'prepare_stock_sequences',
    'plot_training_history',
    'plot_stock_predictions',
    'calculate_metrics',
    'plot_text_generation_sample',
    'plot_feature_importance'
]