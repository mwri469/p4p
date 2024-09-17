import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import sys

class Predictor:
    def __init__(self, dataframe: pd.DataFrame, feature_columns: list, target_columns: list, past=7, future=5, split_fraction=0.715, in_jupyter=False):
        self.show_plots = False
        self.dataframe = dataframe
        self.feature_columns = feature_columns
        self.target_columns = target_columns  # List of target columns
        self.past = past
        self.future = future
        self.split_fraction = split_fraction
        self.scaler = MinMaxScaler()
        self.models = {}  # Dictionary to store models for each target
        self.in_jupyter = in_jupyter  # Attribute to handle Jupyter environment

        # Prepare the data
        self._prepare_data()

        # Build models for each target
        self._build_models()

    def _prepare_data(self):
        # Extract features and targets
        features = self.dataframe[self.feature_columns].values
        self.targets = {tgt: self.dataframe[tgt].values for tgt in self.target_columns}

        # Normalize the features
        self.features_scaled = self.scaler.fit_transform(features)

        # Calculate start and end for training
        start = self.past + self.future
        end = len(features) - self.future  # Ensure we have enough data for future prediction

        # Prepare datasets for each target
        self.X, self.y = {}, {}
        self.train_split = {}
        for tgt in self.target_columns:
            X, y = self._create_dataset(self.features_scaled, self.targets[tgt], self.past, self.future)
            self.X[tgt], self.y[tgt] = X, y
            self.train_split[tgt] = int(self.split_fraction * len(X))

        self.train_dataset, self.val_dataset = {}, {}
        for tgt in self.target_columns:
            X_train, X_val = self.X[tgt][:self.train_split[tgt]], self.X[tgt][self.train_split[tgt]:]
            y_train, y_val = self.y[tgt][:self.train_split[tgt]], self.y[tgt][self.train_split[tgt]:]

            self.train_dataset[tgt] = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(256).prefetch(tf.data.experimental.AUTOTUNE)
            self.val_dataset[tgt] = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(256).prefetch(tf.data.experimental.AUTOTUNE)

    def _create_dataset(self, features, target, past, future):
        X, y = [], []
        for i in range(len(features) - past - future + 1):
            X.append(features[i:i + past])
            y.append(target[i + past:i + past + future])
        return np.array(X), np.array(y)

    def _build_models(self):
        for tgt in self.target_columns:
            model = Sequential([
                LSTM(32, input_shape=(self.past, self.X[tgt].shape[2])),
                Dense(self.future)
            ])
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')
            
            # Do not run model.summary() if in_jupyter is True
            if self.in_jupyter:
                model.summary()
            
            self.models[tgt] = model

    def train(self, epochs=10):
        for tgt, model in self.models.items():
            print(f"Training model for {tgt}...")
            es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
            modelckpt_callback = keras.callbacks.ModelCheckpoint(
                f'{tgt}_model_checkpoint.weights.h5',
                monitor='val_loss',
                save_weights_only=True,
                save_best_only=True
            )
            history = model.fit(
                self.train_dataset[tgt],
                epochs=epochs,
                validation_data=self.val_dataset[tgt],
                callbacks=[es_callback, modelckpt_callback]
            )
            self.models[tgt] = model
            self.history = history

    def visualise_loss(self, title="Training and Validation Loss"):
        if not self.in_jupyter:
            return  # Skip visualizing if running in Jupyter
        plt.figure(figsize=(12, 8))
        for tgt in self.target_columns:
            loss = self.models[tgt].history.history['loss']
            val_loss = self.models[tgt].history.history['val_loss']
            epochs = range(len(loss))
            plt.plot(epochs, loss, 'b', label=f'Training loss ({tgt})')
            plt.plot(epochs, val_loss, 'r', label=f'Validation loss ({tgt})')
        plt.title(title)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def validate(self):
        mse_scores = {}
        for tgt, model in self.models.items():
            mse = []
            for x, y in self.val_dataset[tgt]:
                x_np = x.numpy()
                y_np = y.numpy()
                y_pred = model.predict(x_np)
                mse.append(mean_squared_error(y_np, y_pred))
            mse_scores[tgt] = np.mean(mse)  # Mean MSE across batches
        return mse_scores

    def predict(self, new_data: np.array):
        # Rescale new data
        new_data_scaled = self.scaler.transform(new_data)

        # Reshape new data to match the input shape expected by the LSTM model
        new_data_scaled = np.reshape(new_data_scaled, (1, self.past, len(self.feature_columns)))

        predictions = {}
        for tgt, model in self.models.items():
            pred = model.predict(new_data_scaled)
            predictions[tgt] = pred
        return predictions

    def plot(self, predictions, time_step=5, title="Multi-Feature Prediction"):
        labels = ["History", "True Future", "Model Prediction"]
        marker = [".-", "rx", "go"]

        for tgt, preds in predictions.items():
            plt.figure(figsize=(12, 8))
            plt.title(f"{title} ({tgt})")
            for x_np, y_np, y_pred in preds:
                time_steps = list(range(-(x_np.shape[1]), 0))
                future_steps = list(range(time_step))

                plt.plot(time_steps, x_np[0][:, 0], marker[0], label=f"{labels[0]} ({tgt})")  # Plotting history
                plt.plot(future_steps, y_np[0], marker[1], markersize=10, label=f"{labels[1]} ({tgt})")  # True future
                plt.plot(future_steps, y_pred[0], marker[2], markersize=10, label=f"{labels[2]} ({tgt})")  # Model prediction
                plt.legend()
                plt.xlabel("Time-Step")
                plt.ylabel(tgt)
                plt.show()

# Example usage:
def main():
    # Load your data
    df = pd.read_csv('../data/fulldata.csv')

    # Define feature and target columns
    feature_columns = [
        'WDir(Deg)', 'WSpd(m/s)', 'GustSpd(m/s)', 'WindRun(Km)', 'Rain(mm)', 
        'Tdry(C)', 'RH(%)', 'Tmax(C)', 'Tmin(C)', 
        'Pstn(hPa)', 'Sun(Hrs)', 'Rad(MJ/m2)'
    ]
    target_columns = ['Rain(mm)', 'GustSpd(m/s)', 'Pstn(hPa)', 'Sun(Hrs)', 'Rad(MJ/m2)']  # Predict multiple features

    # Create Predictor instance with the in_jupyter parameter
    predictor = Predictor(df, feature_columns, target_columns, in_jupyter=False)

    # Train the models
    predictor.train(epochs=10)

    # Visualize the loss
    #predictor.visualise_loss()

    # Perform validation and get MSE
    mse_scores = predictor.validate()
    print("Validation MSE scores:", mse_scores)

    # Predict on new data
    new_observation = np.array([[20, 5, 7, 50, 0.2, 25, 80, 32, 21, 1015, 8, 22],
                                [18, 6, 8, 48, 0.1, 24, 82, 31, 20, 1014, 7, 21],
                                [19, 7, 6, 51, 0.15, 23, 81, 30, 19, 1013, 9, 23], 
                                [20, 5, 7, 50, 0.2, 25, 80, 32, 21, 1015, 8, 22],
                                [20, 5, 7, 50, 0.2, 25, 80, 32, 21, 1015, 8, 22],
                                [18, 6, 8, 48, 0.1, 24, 82, 31, 20, 1014, 7, 21],
                                [19, 7, 6, 51, 0.15, 23, 81, 30, 19, 1013, 9, 23]]) 
    predictions = predictor.predict(new_observation)
    print("Predictions:", predictions)

if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
