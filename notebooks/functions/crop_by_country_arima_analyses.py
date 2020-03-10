# This file read in the trade matrix file, takes input for variables 'item' and 'element',
# pre-process data and select countries with have >50% non-zero data points,
# then train an ARIMA model via gridsearch for each country,
# returns best parameter estimates
# and have options to return plots of mean squared error by countries.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# For ARIMA model
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf



# Function for step 1: extract data

def item_element_country(df, item, element):
    '''function to take input of an item name, element name (e.g. Export/import value) and
    output a formated file (aggregated by Reporter Countries) for downstream analysis'''

    data = df.loc[(df['Item'] == item) & (df['Element'] == element), :]
    data_50 = data.loc[data['NoneZero'] >= 16, :]
    print('{} country names selected for those with >50% non-zero data'.format(data_50.shape[0]))

    # Reshape data from wide to long by years
    data_long = data_50.melt(['Reporter Countries'], years, 'year', 'value')

    # Convert data to time series
    data_long['year'] = data_long['year'].map(
        lambda x: x.lstrip('Y'))  # strip Y from year names for easy converting to ts
    data_long.year = pd.to_datetime(data_long.year)

    # Reshape data from long to wide, turn countries into columns
    data_wide = data_long.pivot(index='year', columns='Reporter Countries', values='value')

    return data_wide


# Function for step 2: normalize data###
def df_normalize(df_original):
    '''function to transform and normalize crop trade data
    '''
    # calculate 3-year rolling mean to smooth data and add 1 to prep for log transformation
    # Assume that adding 1 tonne per year will not significantly change the time series patterns for any country
    rolled = df_original.rolling(3).sum().add(1)
    # Log-transformation to scale down countries with large numbers
    # Then divide each data point by the third row's value (removed first two rows which are NaNs)
    # So that all countries have the same start point 0
    df_normalized = np.log(rolled[2:].div(rolled[2:].iloc[0]))

    return df_normalized

# Function for step 3: check stationarity and remove non-stationary countries
def stationarity(df):
    '''function to run Dickey-Fuller test on time series for each country,
    return list of countries that passed the test'''
    non_stationary_countries = []

    for country in df.columns:
        stationarity = adfuller(df[country].dropna())

        if stationarity[1] <= 0.05:
            non_stationary_countries.append(country)
        stationary_countries = [i for i in df.columns.tolist() if i not in non_stationary_countries]
    print('There were {} non-stationary countries being removed and\n result in {} stationary countries'.format(
        len(non_stationary_countries),
        len(stationary_countries)))
    return stationary_countries


# Functions for Step 4: Training an ARIMA model for each country and use gridsearch to find best parameters
# ref: https://machinelearningmastery.com/grid-search-arima-hyperparameters-with-python/

# Step 1): Evaluate an ARIMA model use MSE
def eval_arima(data, order):
    '''function to split a time series into train and test set, build an ARIMA model
      use rolling one-at-a-time predictions, and reture mean squared error of the model'''

    # split dataset into training and testing
    train_size = int(len(data) * 0.66)
    train, test = data[0: train_size], data[train_size:]

    # Make one prediction/forecast at a time, put the prediction into predictions list,
    # then add one data from test set to train set for the next model
    rolling_train = [train_index for train_index in train]
    predictions = []
    for test_index in range(len(test)):
        model = ARIMA(rolling_train, order=order)
        model_fit = model.fit()
        one_prediction = model_fit.forecast()[0]
        predictions.append(one_prediction)
        rolling_train.append(test[test_index])
    mse = mean_squared_error(test, predictions)
    return mse


# Step 2): Grid search for order(p,d,q) of ARIMA model
p_values = [0, 1, 2, 3, 4, 5, 6]
d_values = [0, 1]
q_values = [0, 1, 2]
import warnings
warnings.filterwarnings("ignore")


def gridsearch_arima(data, p_values, d_values, q_values):
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p, d, q)
                try:
                    mse = eval_arima(data, order)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                    # print('ARIMA%s MSE=%.2E' %(order, mse)) # print out results for each configuration
                except:
                    continue
    print(data.name, best_cfg, 'MSE=%.2E' % best_score)
    return data.name, best_cfg, best_score


# Step 3): Loop through all countries to make predictions
def countries_arima(data):
    results = []
    for country in data.columns:
        country_result = gridsearch_arima(data[country], p_values, d_values, q_values)
        results.append(country_result)
    results = pd.DataFrame(results, columns=['country', 'best_arima', 'mse'])
    return results

########################################################################

# Putting all together:

def crop_country_arima_analysis(data, item, element, plot=True):
    '''function to conduct crop-by-country arima analysis'''

    # step 1 : extract data
    df_original = item_element_country(data, item, element)

    # step 2 : normalize data
    df_normalized = df_normalize(df_original)

    # Plotting 1
    if plot==True:
        # Visually compare original data and normalized data
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        ax1.plot(df_original)
        ax2.plot(df_normalized)

        ax1.set_title('original')
        ax1.set_ylabel('tonnes')
        ax2.set_title('normalized')
        ax2.set_ylabel('normalized scale (log-transformed)')
        fig.suptitle('Annual {} {} by Countries'.format(item, element))
        plt.show()

        import warnings
        warnings.filterwarnings("ignore")

    # Step 3: check stationarity and remove non-stationary countries
    stationary_countries = stationarity(df_normalized) # Check stationarity

    df_processed = df_normalized[stationary_countries] # Remove non-stationary countries from data frame

    # Step 5 Training an ARIMA model for each country and use gridsearch to find best parameters
    df_arima = countries_arima(df_processed)

    # Save results
    df_arima = df_arima.set_index('country')
    df_arima.to_csv('../data/processed/arima_{}_{}_.csv'.format(item, element))

    # Plotting 2: MSE
    if plot == True:

        df_arima = df_arima.sort_values(by='mse')
        plt.figure(figsize=(20, 10))
        plt.plot(df_arima['mse'])
        plt.xlabel('Country', fontsize=20)
        plt.ylabel('Mean squared error of best ARIMA model', fontsize=20)
        plt.xticks(fontsize=12, rotation=90)
        plt.yticks(fontsize=14)
        plt.show()