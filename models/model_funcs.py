import keras
from keras.models import Sequential, Model # type: ignore
from keras.layers import LSTM, Dense, Bidirectional, Conv1D, MaxPooling1D, Dropout, BatchNormalization # type: ignore
from keras.layers import Multiply, Flatten, Activation, RepeatVector, Permute, Input, Add, GRU # type: ignore
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

def two_layer_bidirectional_lstm(past, future, input_shape):
    """ Two Bidirectional LSTM layers stacked with Dense output. """
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True, input_shape=(past, input_shape[2]))),
        Bidirectional(LSTM(32)),
        Dense(future)
    ])
    return model

def deep_bidirectional_lstm_with_dropout(past, future, input_shape):
    """ Deep Bidirectional LSTM model with dropout after each LSTM layer. """
    model = Sequential([
        Bidirectional(LSTM(128, return_sequences=True, input_shape=(past, input_shape[2]))),
        Dropout(0.3),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.3),
        Bidirectional(LSTM(32)),
        Dense(future)
    ])
    return model

def bidirectional_lstm_with_recurrent_dropout(past, future, input_shape):
    """ Bidirectional LSTM with recurrent dropout inside the LSTM layers. """
    model = Sequential([
        Bidirectional(LSTM(128, recurrent_dropout=0.2, return_sequences=True, input_shape=(past, input_shape[2]))),
        Bidirectional(LSTM(64, recurrent_dropout=0.2)),
        Dense(future)
    ])
    return model

def bidirectional_lstm_with_dense_layers(past, future, input_shape):
    """ Bidirectional LSTM followed by multiple dense layers. """
    model = Sequential([
        Bidirectional(LSTM(128, return_sequences=True, input_shape=(past, input_shape[2]))),
        Bidirectional(LSTM(64)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(future)
    ])
    return model

def cnn_bidirectional_lstm_hybrid(past, future, input_shape):
    """ CNN layer followed by Bidirectional LSTMs for hybrid sequence modeling. """
    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=(past, input_shape[2])),
        MaxPooling1D(pool_size=2),
        Bidirectional(LSTM(128, return_sequences=True)),
        Bidirectional(LSTM(64)),
        Dense(future)
    ])
    return model

def very_deep_bidirectional_lstm(past, future, input_shape):
    """ Very deep Bidirectional LSTM model with four layers and dropout. """
    model = Sequential([
        Bidirectional(LSTM(256, return_sequences=True, input_shape=(past, input_shape[2]))),
        Dropout(0.3),
        Bidirectional(LSTM(128, return_sequences=True)),
        Dropout(0.3),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.3),
        Bidirectional(LSTM(32)),
        Dense(future)
    ])
    return model

def bidirectional_lstm_with_batch_norm(past, future, input_shape):
    """ Bidirectional LSTM layers with Batch Normalization. """
    model = Sequential([
        Bidirectional(LSTM(128, return_sequences=True, input_shape=(past, input_shape[2]))),
        BatchNormalization(),
        Bidirectional(LSTM(64, return_sequences=True)),
        BatchNormalization(),
        Bidirectional(LSTM(32)),
        Dense(future)
    ])
    return model

def bidirectional_lstm_with_attention(past, future, input_shape):
    """ Bidirectional LSTM with attention mechanism using Keras Functional API. """

    # Attention mechanism
    def attention_layer(inputs):
        attention = Dense(1, activation='tanh')(inputs)
        attention = Flatten()(attention)
        attention = Activation('softmax')(attention)
        attention = RepeatVector(inputs.shape[-1])(attention)
        attention = Permute([2, 1])(attention)
        output = Multiply()([inputs, attention])
        return output

    # Input layer
    input_layer = Input(shape=(past, input_shape[2]))

    # Bidirectional LSTM layers
    x = Bidirectional(LSTM(128, return_sequences=True))(input_layer)
    
    # Apply attention
    x = attention_layer(x)

    # Another Bidirectional LSTM layer after attention
    x = Bidirectional(LSTM(64))(x)

    # Output layer
    output_layer = Dense(future)(x)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

def bidirectional_lstm_with_residual_connections(past, future, input_shape):
    """ Bidirectional LSTM with residual connections between layers. """
    
    input_layer = Input(shape=(past, input_shape[2]))
    
    # First LSTM layer
    x = Bidirectional(LSTM(128, return_sequences=True))(input_layer)
    
    # Residual connection
    residual = x
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = Add()([x, residual])  # Residual connection
    
    # Output
    x = Bidirectional(LSTM(32))(x)
    output_layer = Dense(future)(x)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

def bidirectional_lstm_with_gru(past, future, input_shape):
    """ Bidirectional LSTM followed by GRU layer. """
    model = Sequential([
        Bidirectional(LSTM(128, return_sequences=True, input_shape=(past, input_shape[2]))),
        GRU(64, return_sequences=True),
        Bidirectional(LSTM(32)),
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