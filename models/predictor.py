import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import keras
from keras.models import Sequential # type: ignore
from keras.layers import LSTM, Dense # type: ignore
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
        self.scalerY = {tgt: MinMaxScaler() for tgt in self.target_columns}
        self.models = {}  # Dictionary to store models for each target
        self.in_jupyter = in_jupyter  # Attribute to handle Jupyter environment

        if not in_jupyter: sys.stdout.reconfigure(encoding='utf-8') # allows for utf-8 encoding in terminal

        # Prepare the data
        self._prepare_data()

        # Build models for each target
        self._build_models()

    def _prepare_data(self):
        # Extract features and targets
        features = self.dataframe[self.feature_columns].values
        self.targets = {tgt: self.scalerY[tgt].fit_transform(self.clean_numpy(self.dataframe[tgt].values).reshape(-1, 1)) for tgt in self.target_columns}

        # Normalise the features
        features = self.clean_numpy(features)
        self.features_scaled = self.scaler.fit_transform(features)

        # Prepare datasets for each target
        self.X, self.y = {}, {}
        self.train_split = {}
        for tgt in self.target_columns:
            X, y = self._create_dataset(self.features_scaled, self.targets[tgt], self.past, self.future)
            self.X[tgt], self.y[tgt] = X, y
            self.train_split[tgt] = int(self.split_fraction * len(X))

        self.train_dataset, self.test_dataset = {}, {}
        for tgt in self.target_columns:
            X_train, X_val = np.asarray(self.X[tgt][:self.train_split[tgt]]).astype('float64'), np.asarray(self.X[tgt][self.train_split[tgt]:]).astype('float64')
            y_train, y_val = np.asarray(self.y[tgt][:self.train_split[tgt]]).astype('float64'), np.asarray(self.y[tgt][self.train_split[tgt]:]).astype('float64')

            self.train_dataset[tgt] = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(256).prefetch(tf.data.experimental.AUTOTUNE)
            self.test_dataset[tgt] = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(256).prefetch(tf.data.experimental.AUTOTUNE)

    def clean_numpy(self, numpy_arr):
        """ Cleans and imputes numpy arrays """
        for obs in range(len(numpy_arr)):
            # Check for NaN values
            if not self.is_array_like(numpy_arr[0]):
                if numpy_arr[obs] == '-':
                    if obs == 0:
                        numpy_arr[obs] = np.float64(0)
                    else:
                        numpy_arr[obs] = np.float64(numpy_arr[obs-1])
                else:
                    numpy_arr[obs] = np.float64(numpy_arr[obs])
            else:
                for feat in range(len(numpy_arr[0])):
                    if numpy_arr[obs,feat] == '-':
                        if obs == 0:
                            numpy_arr[obs,feat] = np.float64(0)
                        else:
                            numpy_arr[obs,feat] = numpy_arr[obs-1,feat]
                    else:
                        numpy_arr[obs,feat] = np.float64(numpy_arr[obs,feat])

        return numpy_arr
    
    def is_array_like(self, obj):
        """Checks if an object is array-like.
        Args:
            obj: The object to check.
        Returns:
            True if the object is array-like, False otherwise.
        """

        return isinstance(obj, np.ndarray) or hasattr(obj, '__iter__') and not isinstance(obj, str)

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
            print(self.train_dataset[tgt])
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
                validation_data=self.test_dataset[tgt],
                callbacks=[es_callback, modelckpt_callback]
            )
            self.models[tgt] = model
            self.history = history

    def visualise_loss(self, title="Training and Validation Loss"):
        if self.in_jupyter:
            return
        plt.figure(figsize=(12, 8))
        for tgt in self.target_columns:
            loss = self.models[tgt].history.history['loss']
            val_loss = self.models[tgt].history.history['val_loss']
            epochs = range(len(loss))
            #plt.plot(epochs, loss, 'b', label=f'Training loss ({tgt})')
            plt.plot(epochs, val_loss, 'r', label=f'Validation loss ({tgt})')
        plt.title(title)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def validate(self, val_data):
        # Use the trained scaler to transform the validation features
        X_val = self.scaler.transform(self.clean_numpy(val_data[self.feature_columns].values))
        
        # Prepare the target data for each target column
        y_val = {tgt: self.scalerY[tgt].transform(self.clean_numpy(val_data[tgt].values).reshape(-1, 1)) for tgt in self.target_columns}
        
        # Reshape the feature data for LSTM input (batch_size, timesteps, num_features)
        X_val = np.array([X_val[i:i + self.past] for i in range(len(X_val) - self.past)])

        mse_scores = {}

        # Loop through each target column and corresponding model
        for tgt, model in self.models.items():
            # Prepare the target data for validation (shifted to match the future horizon)
            # Make sure we don't go out of bounds when selecting future steps
            y_tgt = np.array([
                y_val[tgt][i + self.past:i + self.past + self.future].flatten() 
                for i in range(len(y_val[tgt]) - self.past - self.future + 1)
            ])
            
            # Get predictions from the model
            y_pred = model.predict(X_val[:len(y_tgt)])  # Ensure the same length for X_val and y_tgt
            
            # Calculate MSE between predictions and actual targets
            mse = mean_squared_error(y_tgt, y_pred)
            mse_scores[tgt] = mse

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
    df = pd.read_csv('./data/fulldata.csv')

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
    predictor.visualise_loss()

    # Perform validation and get MSE
    # mse_scores = predictor.validate()
    # print("Validation MSE scores:", mse_scores)

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
