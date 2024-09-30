import keras
from keras.models import Sequential # type: ignore
from keras.layers import LSTM, Dense, Bidirectional, Conv1D, MaxPooling1D, Dropout # type: ignore
TF_ENABLE_ONEDNN_OPTS=0

def create_custom_model(s_past, s_future, s_X_shape):
    """ Example of a more complex architecture with multiple LSTM layers and dropout.
    You can add different architectures here and pass them via the options dictionary."""
    
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(s_past, s_X_shape[2])),
        LSTM(32),
        Dropout(0.2),
        Dense(s_future)
    ])
    return model

def single_lstm_dense_output(past, future, input_shape):
    """ Single LSTM layer with a Dense output layer. """
    model = Sequential([
        LSTM(32, input_shape=(past, input_shape[2])),
        Dense(future)
    ])
    return model

def two_layer_lstm(past, future, input_shape):
    """ Two stacked LSTM layers with a Dense output layer. """
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(past, input_shape[2])),
        LSTM(32),
        Dense(future)
    ])
    return model

def lstm_with_dropout(past, future, input_shape):
    """ LSTM with Dropout for regularization to prevent overfitting. """
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(past, input_shape[2])),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.3),
        Dense(future)
    ])
    return model

def bidirectional_lstm(past, future, input_shape):
    """ Bidirectional LSTM layer followed by Dense output. """
    model = Sequential([
        Bidirectional(LSTM(64, input_shape=(past, input_shape[2]))),
        Dense(future)
    ])
    return model

def lstm_with_recurrent_dropout(past, future, input_shape):
    """ LSTM with recurrent dropout inside the LSTM units. """
    model = Sequential([
        LSTM(64, input_shape=(past, input_shape[2]), recurrent_dropout=0.2),
        Dense(future)
    ])
    return model

def deep_lstm_with_dense_layers(past, future, input_shape):
    """ Deep LSTM architecture with multiple Dense layers. """
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(past, input_shape[2])),
        LSTM(32),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(future)
    ])
    return model

def cnn_lstm_hybrid(past, future, input_shape):
    """ CNN followed by LSTM for hybrid feature extraction and sequence modeling. """
    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=(past, input_shape[2])),
        MaxPooling1D(pool_size=2),
        LSTM(64),
        Dense(future)
    ])
    return model

def shallow_lstm_linear_output(past, future, input_shape):
    """ Shallow LSTM with a linear output activation (used for regression). """
    model = Sequential([
        LSTM(32, input_shape=(past, input_shape[2])),
        Dense(future, activation='linear')
    ])
    return model