"""
This code is used to compare two locations
"""
from predictor import Predictor
import numpy as np
import pandas as pd

def fitness(day_prediction):
    """Fitness function to evaluate weather predictions between two locations, loc1 and loc2.
    Inputs: passed in as a dictionary <- {'Rain(mm)': [loc1, loc2], 'GustSpd(m/s)': [loc1, loc2], 'Pstn(Pstn(hPa)': ...,
                                            'Sun(Hrs)': ..., 'Rad(MJ/m2)': [loc1, loc2]}"""
    # Define the different parameters and their weightings
    params = ['Rain(mm)', 'GustSpd(m/s)', 'Pstn(hPa)', 'Rad(MJ/m2)']
    weights = [0.3, 0.3, 0.1, 0.2]

    # Invert station pressure variable as higher pressure is more likely to mean better weather
    day_prediction['Pstn(hPa)'] = (-1)*day_prediction['Pstn(hPa)']

    # Calculate a score for both locations
    loc1, loc2 = 0,0

    for idx in range(len(params)):
        loc1 += weights[idx]*day_prediction[params[idx]][0]
        loc2 += weights[idx]*day_prediction[params[idx]][1]

    # Return index of better location
    if loc1 <= loc2:
        return 0
    else:
        return 1

def main():
    pass

if __name__ == '__main__':
    main()