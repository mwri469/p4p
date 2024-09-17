import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

class Predictor:
    def __init__(self, dataframe, predict_idx):
        pass

    def predict(self):
        pass

    def plot(self, time_step, title):
        pass

# Load your data
df = pd.read_csv('fulldata.csv')

# Feature columns as per your request
feature_columns = [
    'WDir(Deg)', 'WSpd(m/s)', 'WindRun(Km)', 'Rain(mm)', 
    'Tdry(C)', 'RH(%)', 'Tmax(C)', 'Tmin(C)', 'ET05(C)', 
    'Pstn(hPa)', 'Sun(Hrs)', 'Rad(MJ/m2)'
]

# Selecting features and target columns
features = df[feature_columns].values
target = df['Tdry(C)'].values

# Normalize the data
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

# Parameters
past = 7  # past 7 days
future = 5  # predict 5 days into the future
sequence_length = past
batch_size = 256
epochs = 10

# Prepare data for time series dataset
def create_dataset(features, target, past, future):
    X, y = [], []
    for i in range(len(features) - past - future + 1):
        X.append(features[i:i+past])
        y.append(target[i+past:i+past+future])
    return np.array(X), np.array(y)

X, y = create_dataset(features_scaled, target, past, future)

# Split into train and validation sets
split_fraction = 0.715
train_split = int(split_fraction * len(X))

X_train, X_val = X[:train_split], X[train_split:]
y_train, y_val = y[:train_split], y[train_split:]

# Create Keras datasets
train_dataset = keras.preprocessing.timeseries_dataset_from_array(
    X_train,
    y_train,
    sequence_length=sequence_length,
    batch_size=batch_size,
)

val_dataset = keras.preprocessing.timeseries_dataset_from_array(
    X_val,
    y_val,
    sequence_length=sequence_length,
    batch_size=batch_size,
)

# Define the LSTM model
model = Sequential([
    LSTM(32, input_shape=(sequence_length, X_train.shape[2])),
    Dense(future)
])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')
model.summary()

# Train the model
es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
modelckpt_callback = keras.callbacks.ModelCheckpoint(
    'model_checkpoint.weights.h5',
    monitor='val_loss',
    save_weights_only=True,
    save_best_only=True
)

history = model.fit(
    train_dataset,
    epochs=epochs,
    validation_data=val_dataset,
    callbacks=[es_callback, modelckpt_callback]
)

# Visualization of loss
def visualize_loss(history, title):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

visualize_loss(history, 'Training and Validation Loss')

# Function to plot predictions
def show_plot(plot_data, delta, title):
    labels = ["History", "True Future", "Model Prediction"]
    marker = [".-", "rx", "go"]
    time_steps = list(range(-(plot_data[0].shape[0]), 0))
    future_steps = list(range(delta))

    plt.title(title)
    for i, val in enumerate(plot_data):
        if i == 1:  # True Future
            plt.plot(future_steps, val, marker[i], markersize=10, label=labels[i])
        elif i == 2:  # Model Prediction
            plt.plot(future_steps, val, marker[i], markersize=10, label=labels[i])
        else:
            plt.plot(time_steps, val.flatten(), marker[i], label=labels[i])
    plt.legend()
    plt.xlim([time_steps[0], future_steps[-1] * 2])
    plt.xlabel("Time-Step")
    plt.show()
    return

# Predict and plot
for x, y in val_dataset.take(5):
    x_np = x.numpy()
    y_np = y.numpy()
    y_pred = model.predict(x_np)
    show_plot([x_np[0][:, 0], y_np[0], y_pred[0]], future, "Single Step Prediction")