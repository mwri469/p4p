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

class helperfuncs:
    def prepare(self, df, feats, tgts):
        features = df[feats].values
        targets = {tgt: self.clean_numpy(df[tgt].values) for tgt in tgts}

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