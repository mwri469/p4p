import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from predictor import Predictor, DataProcessor
from comparer import fitness

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

def eval_locs_on_comp(file1_name, file2_name):
    # Loading in two datasets of the same timespan
    loc1 = pd.read_csv(file1_name)
    loc2 = pd.read_csv(file2_name)

    loc1_process = DataProcessor(loc1)
    loc2_process = DataProcessor(loc2)

    loc1_process.dt_col()
    loc2_process.dt_col()

    assert loc1_process.max_date == loc2_process.max_date

    loc1_test, loc1_val = loc1_process.test_val_split(num_months=12)
    loc2_test, loc2_val = loc2_process.test_val_split(num_months=12)

    loc1_model = Predictor(loc1_test, FEATURES, TARGETS)
    loc2_model = Predictor(loc2_test, FEATURES, TARGETS)

    print(loc1_val.head())
    for obs in loc1_val:
        print(obs)

def main():
    f1 = './data/motat020316_010924.csv'
    f2 = './data/leigh020316_010924.csv'
    eval_locs_on_comp(f1, f2)


def main0():
    loc1_data = pd.read_csv('./data/motat020316_010924.csv')
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