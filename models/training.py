import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from predictor import Predictor, DataProcessor
from comparer import fitness
import keras
from keras.models import Sequential # type: ignore
from keras.layers import LSTM, Dense # type: ignore
from model_funcs import *
import tensorflow as tf
tf.keras.config.disable_interactive_logging()
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

# Define feature and target columns
FEATURES = [
    'WDir(Deg)', 'WSpd(m/s)', 'GustSpd(m/s)', 'WindRun(Km)', 'Rain(mm)', 
    'Tdry(C)', 'RH(%)', 'Tmax(C)', 'Tmin(C)', 
    'Pstn(hPa)', 'Rad(MJ/m2)'
]
TARGETS = ['Rain(mm)', 'GustSpd(m/s)', 'Pstn(hPa)', 'Rad(MJ/m2)']  # Predict multiple features


def eval_over_time(df, num_months):
    """ Plots the performance of the predictor based on how many months' data
    it trains on"""
    assert num_months > 3 # Check enough data is being passed in

    months = [x for x in range(num_months)]
    mse_performance = []

    # Bit of preproccess
    df['Day(Local_Date)'] = pd.to_datetime(df['Day(Local_Date)'], format='%Y%m%d:%H%M')
    
    start_train = df['Day(Local_Date)'].min()
    start_val = df['Day(Local_Date)'].max() - pd.DateOffset(months=3)

    validation_data = df.loc[(df['Day(Local_Date)'].dt.year >= start_val.year) &
                            (df['Day(Local_Date)'].dt.month >= start_val.month)]
    # print(validation_data[0:5])
    # val = Predictor(validation_data, FEATURES, TARGETS) # Just to get filtered data

    for month in months[2:-3]:
        # Last 3 months is validation data
        # This line filters data down to right range
        end_point = start_train + pd.DateOffset(months=month)
        print(f'Training on current date: {end_point.month}, {end_point.year} . . .')
        df_months_filtered = df.loc[(df['Day(Local_Date)'].dt.year <= end_point.year) &
                                    (df['Day(Local_Date)'].dt.month <= end_point.month)]
        # print(df_months_filtered[0:10])
        m = Predictor(df_months_filtered, FEATURES, TARGETS)
        m.train(epochs=12)
        mse_performance.append(np.mean(list(m.validate(validation_data).values())))

    
    pd.DataFrame({'Month': months[2:-3], 'MSE': mse_performance}).to_csv('validation.csv')
    print('months', months)
    print('mse', mse_performance)
    plt.plot(months[2:-3], mse_performance)
    plt.title('No. months of data input vs. model performance of validation dataset')
    plt.ylabel('MSE Validation performance')
    plt.xlabel('Number of months of data trained on')
    plt.show()

def seq_model(self, s_past, s_future, s_X, s_tgt):
    return Sequential([
                LSTM(32, input_shape=(s_past, s_X[s_tgt].shape[2])),
                Dense(s_future)
                    ])

def eval_locs_on_comp(file1_name, file2_name, options):
    # Loading in two datasets of the same timespan
    loc1 = pd.read_csv(file1_name)
    loc2 = pd.read_csv(file2_name)

    processor = DataProcessor()

    # Process the data
    loc1_process = DataProcessor()
    loc1_process.optional__init__(loc1)
    loc2_process = DataProcessor()
    loc2_process.optional__init__(loc2)

    # Assuming dt_col handles date-time column processing
    loc1_process.dt_col()
    loc2_process.dt_col()

    # Ensure both locations have data for the same timespan
    assert loc1_process.max_date == loc2_process.max_date

    # Split into test and validation sets
    loc1_test, loc1_val = loc1_process.test_val_split(num_months=12)
    loc2_test, loc2_val = loc2_process.test_val_split(num_months=12)

    # Initialize the models
    loc1_model = Predictor(loc1_test, FEATURES, TARGETS, using_options=True, options=options)
    loc2_model = Predictor(loc2_test, FEATURES, TARGETS, using_options=True, options=options)

    # Prepare results tracking
    correct_predictions = 0
    total_comparisons = 0

    # Iterate over validation data
    for i in range(len(loc1_val) - loc1_model.past - loc1_model.future + 1):
        # Get past 7 days (loc1_model.past) of data for both locations
        loc1_past_data = loc1_val[loc1_model.feature_columns].iloc[i:i + loc1_model.past].values
        loc2_past_data = loc2_val[loc2_model.feature_columns].iloc[i:i + loc2_model.past].values

        # Ensure the data is non-normalized and is in the correct shape (7 days of data)
        loc1_past_data = np.array(loc1_past_data)
        loc2_past_data = np.array(loc2_past_data)

        # Predict future weather conditions (next 5 days)
        loc1_predictions = loc1_model.predict(loc1_past_data)
        loc2_predictions = loc2_model.predict(loc2_past_data)

        # Compare predictions using comparer()
        # Assume comparer() takes a dictionary like {'Rain(mm)': [loc1, loc2], 'GustSpd(m/s)': [loc1, loc2], ...}
        comparison_inputs = {
            feature: [loc1_predictions[feature][0][0], loc2_predictions[feature][0][0]]  # Compare only the first predicted day
            for feature in TARGETS
        }
        
        # comparer() returns 0 if loc1 is better, 1 if loc2 is better
        better_location = fitness(comparison_inputs)

        # Get the actual future weather (next 5 days) for both locations
        loc1_actual = loc1_val[TARGETS].iloc[i + loc1_model.past:i + loc1_model.past + loc1_model.future].values
        loc2_actual = loc2_val[TARGETS].iloc[i + loc2_model.past:i + loc2_model.past + loc2_model.future].values

        # Compare the actual better location using the same comparer logic
        # Use the scalerY from trained model to normalise data and match the model testing
        actual_inputs = {
            feature: np.array([loc1_model.scalerY[feature].transform(
                                np.float32(loc1_actual[0][idx]).reshape((-1,1))
                                ),
                                loc1_model.scalerY[feature].transform(
                                np.float32(loc2_actual[0][idx]).reshape((-1,1))
                                )
                                ], 
                                dtype=np.float32)  # Compare only the first actual day
            for idx, feature in enumerate(TARGETS)
        }

        # Flatten lists
        for feature in TARGETS:
            actual_inputs[feature] = actual_inputs[feature].flatten()

        actual_better_location = fitness(actual_inputs)

        # Increment correct predictions if the model's prediction matches the actual outcome
        if better_location == actual_better_location:
            correct_predictions += 1
        total_comparisons += 1

    # Calculate and print the accuracy of the model in predicting the better location
    accuracy = correct_predictions / total_comparisons
    print(f"Model accuracy in predicting the better location: {accuracy:.2%}")
    return accuracy

def grid_search():
    dp = DataProcessor()

    # Define different model architectures and learning rates for grid search
    models = [
        dp.seq_model,  # Default model
        create_custom_model,  # More complex architecture
        single_lstm_dense_output,
        two_layer_lstm,
        lstm_with_dropout,
        bidirectional_lstm,
        lstm_with_recurrent_dropout,
        deep_lstm_with_dense_layers,
        cnn_lstm_hybrid,
        shallow_lstm_linear_output,
        two_layer_bidirectional_lstm,
        deep_bidirectional_lstm_with_dropout,
        bidirectional_lstm_with_recurrent_dropout,
        bidirectional_lstm_with_dense_layers,
        cnn_bidirectional_lstm_hybrid,
        very_deep_bidirectional_lstm,
        bidirectional_lstm_with_batch_norm,
        bidirectional_lstm_with_attention,
        bidirectional_lstm_with_residual_connections,
        bidirectional_lstm_with_gru
        ]   

    learning_rates = [0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00001]

    df = pd.read_csv('./data/leigh2010_010924.csv')

    dp.optional__init__(df)
    dp.dt_col()
    test, val = dp.test_val_split(num_months=9)

    results = {'Model': [], 'Learning Rate': [], 'MSE': []}

    for md in models:
        for lr in learning_rates:
            print(f'Testing model {md} on learning rate {lr} . . .')
            dp.base_options['model'] = md
            dp.base_options['optimiser'] = dp.opti(lr)

            predictor = Predictor(dataframe=test, feature_columns=FEATURES, target_columns=TARGETS,
                                  using_options=True, options=dp.base_options)
            
            results['Learning Rate'].append(lr)
            results['Model'].append(md)
            results['MSE'].append(np.mean(list(
                predictor.validate(val).values()
                )))
            
    newdf = pd.DataFrame(results)
    newdf.to_csv('./data/grid_search.csv')

def main():
    grid_search()

def main0():
    # Set filepath for data
    f1 = './data/motat020316_010924.csv'
    f2 = './data/leigh020316_010924.csv'

    # Run hyperparameter gridsearch
    options = {
        
    }

    eval_locs_on_comp(f1, f2)


def main1():
    loc1_data = pd.read_csv('./data/leigh2010_010924.csv')
    loc1_data['Day(Local_Date)'] = pd.to_datetime(loc1_data['Day(Local_Date)'], format='%Y%m%d:%H%M')
    n_months = 0
    start = loc1_data['Day(Local_Date)'].min()
    end = loc1_data['Day(Local_Date)'].max()
    month = start.month
    year = start.year
    while True:
        if month == end.month and year == end.year:
            break

        month += 1
        if month >12:
            month = 1
            year += 1

        n_months += 1

    eval_over_time(loc1_data, n_months)

if __name__ == '__main__':
    main()