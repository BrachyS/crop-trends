# This file read in the trade matrix file, takes input for variables 'item' and 'element',
# pre-process data and select countries with have >50% non-zero data points,
# then train an ARIMA model via gridsearch for each country,
# returns best parameter estimates
# and have options to return plots of mean squared error by countries.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# For ARIMA model
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf



# Function for step 1: extract data

def item_element_country(df, years, item, element):
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
    rolled = df_original.rolling(3).mean().add(1)

    # Log-transformation to scale down countries with large numbers
    # Then divide each data point by the third row's value (removed first two rows which are NaNs)
    # So that all countries have the same start point 0
    #df_normalized = np.log(rolled[2:].div(rolled[2:].iloc[0]))
    df_normalized = rolled[2:]
    return df_normalized

# Function for step 3: check stationarity and remove non-stationary countries
def stationarity(df):
    '''function to run A-Dickey-Fuller test on time series for each country,
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
# ref: https://www.machinelearningplus.com/time-series/arima-model-time-series-forecasting-python/
# Step 4-1): Evaluate an ARIMA model use MSE

split = 0.7  # default train-test split value
def eval_arima(data, order, split=split):
    '''function to split a time series into train and test set, build an ARIMA model
      use rolling one-at-a-time predictions, and reture mean squared error of the model'''

    # split dataset into training and testing
    train_size = int(len(data) * split)
    train, test = data[0: train_size], data[train_size:]

    # Make one prediction/forecast at a time, put the prediction into predictions list,
    # then add one data from test set to train set for the next model
    rolling_train = [train_data for train_data in train]
    predictions = []
    for test_data in range(len(test)):
        model = ARIMA(rolling_train, order=order)
        model_fit = model.fit()
        one_prediction = model_fit.forecast()[0]
        predictions.append(one_prediction)
        rolling_train.append(test[test_data])
    predictions = np.squeeze(predictions) # reshape data
    mse = mean_squared_error(test, predictions) # Mean squared error

    mape = np.mean(np.abs(predictions-test)/np.abs(test))  # Mean absolute percentage error

    corr = np.corrcoef(predictions,test)[0,1] # correlation between fitted and test data

    mins = np.amin(np.hstack([predictions[:,None], test[:, None]]), axis=1) # find min values of fitted and test
    maxs = np.amax(np.hstack([predictions[:,None], test[:, None]]), axis=1) # find max values of fitted and test
    minmax = 1 - np.mean(mins/maxs) # Min-max error

    return({'MSE':mse, 'MAPE': mape, 'Corr': corr, 'MinMax': minmax})

# Step 4-2): Grid search for order(p,d,q) of ARIMA model
p_values = [0, 1, 2, 3, 4, 5, 6]
#d_values = [0]  # processed data are non-stationary so no need to optimize d value
q_values = [0, 1, 2]


import warnings
warnings.filterwarnings("ignore")


def gridsearch_arima(data, p_values, q_values, split=split):
    mse, mape, corr, minmax, best_cfg = float("inf"),float("inf"),float("inf"),float("inf"), None
    for p in p_values:
        for q in q_values:
            order = (p, 0, q)
            split = split
            try:
                metrics = eval_arima(data, order, split)
                if metrics['MSE'] < mse: # Use MSE as accuracy metric, find minimum MSE
                    mse, mape, corr, minmax, best_cfg = metrics['MSE'], metrics['MAPE'], metrics['Corr'], metrics['MinMax'], [order[0],order[1],order[2]]
            except:
                continue
    print(data.name, best_cfg, 'MSE=%.2E' % mse, 'MAPE=%.2E' % mape, 'Corr=%.2E' % corr,)
    return data.name, best_cfg, mse, mape, corr, minmax


# Step 4-3): Loop through all countries to make predictions
def countries_arima(data):
    results = []
    for country in data.columns:
        country_result = gridsearch_arima(data[country], p_values, q_values)
        results.append(country_result)
    results = pd.DataFrame(results, columns=['country', 'best_arima', 'mse','mape','corr','minmax'])
    return results

########################################################################

# Putting all together:

# Function 1: data preprocessing and visualization
def crop_country_preprocess(data, years, item, element, plot=False):
    '''function to conduct crop-by-country arima analysis'''

    # step 1 : extract data
    df_original = item_element_country(data, years, item, element)

    if df_original.shape[1] == 0:
        print('Insufficient data for analyzing {}'.format(item))
    else:
        # step 2 : normalize data
        df_normalized = df_normalize(df_original)

        # Step 3: check stationarity and remove non-stationary countries
        stationary_countries = stationarity(df_normalized) # Check stationarity
        df_processed = df_normalized[stationary_countries] # Remove non-stationary countries from data frame

        if df_processed.shape[1] == 0:
            print('Insufficient data for modeling {}'.format(item))
        else:
        # Plotting 1
            if plot==True:
                # Visually compare original data and normalized data
                fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(15, 5))
                ax1.plot(df_original)
                ax2.plot(df_normalized)

                ax1.set_title('original')
                ax1.set_ylabel('tonnes')
                ax2.set_title('Three-year Average')

                fig.suptitle('Annual {} {} by Countries'.format(item, element))
                plt.show()

                import warnings
                warnings.filterwarnings("ignore")

            return df_original, df_processed

# Function 2: ARIMA modeling


def crop_country_arima(df_processed, item, element, plot=False):
    '''function to conduct crop-by-country arima analysis'''

    # Step 5 Training an ARIMA model for each country and use gridsearch to find best parameters
    df_arima = countries_arima(df_processed)

    # Save results
    df_arima = df_arima.set_index('country')
    #df_arima.to_csv('../data/processed/arima/arima_{}_{}_.csv'.format(item, element))

    # Plotting 2: MSE
    # ref: https://matplotlib.org/3.1.0/gallery/subplots_axes_and_figures/subplots_demo.html

    if plot == True:
        df_arima = df_arima.sort_values(by='mape')
        fig, axs = plt.subplots(2, 2, figsize=(20, 12))


        for ax in fig.axes:
            plt.sca(ax)
            plt.xticks(rotation=90)
            axs[0, 0].plot(df_arima['mape'])
            axs[0, 0].set_title('Mean Absolute Percentage Error')
            axs[0, 1].plot(df_arima['corr'])
            axs[0, 1].set_title('Correlation between predictions and test data')
            axs[1, 0].plot(df_arima['minmax'])
            axs[1, 0].set_title('Min-Max Error')
            axs[1, 1].plot(df_arima['mse'])
            axs[1, 1].set_title('Mean Squared Error')

    return df_arima



# Function 3: ARIMA forecasting for each country using best parameters

# default 8 years, into 2025


def crop_country_forecast(df_processed, df_arima, n_periods=8):

    index_future = pd.date_range(start='2018', periods= n_periods, freq='AS-JAN')
    index = pd.date_range(start='1988', end=index_future.tolist()[-1], freq='AS-JAN')

    forecast_values = pd.DataFrame(index=index)
    lower_ci_values = pd.DataFrame()
    upper_ci_values = pd.DataFrame()

    for country in df_processed.columns:
        processed = df_processed[country]

        try:
            model = ARIMA(processed, order=df_arima.loc[country][0])
            model_fit = model.fit()

            # forecast 8 years into the future -- 2025
            forecast = model_fit.predict(start='1988', end=index_future.tolist()[-1])
            forecast.name = country


            # Get confidence intervals
            lower_ci = pd.Series(model_fit.forecast(n_periods)[2][:, 0])
            lower_ci.name = country
            upper_ci = pd.Series(model_fit.forecast(n_periods)[2][:, 1])
            upper_ci.name = country

            forecast_values = pd.concat([forecast_values, pd.DataFrame(forecast)], axis=1)
            lower_ci_values = pd.concat([lower_ci_values, pd.DataFrame(lower_ci)], axis=1)
            upper_ci_values = pd.concat([upper_ci_values, pd.DataFrame(upper_ci)], axis=1)
            # print('{} fitted'.format(country))  # progress check

        except:
            continue

    lower_ci_values = lower_ci_values.set_index(index_future)
    upper_ci_values = upper_ci_values.set_index(index_future)

    return forecast_values, lower_ci_values, upper_ci_values




########################################################################
########################################################################


def country_selected(df, years, item, element, countries):


    data = df.loc[(df['Item'] == item) & (df['Element'] == element), :]

    # Reshape data from wide to long by years
    data_long = data.melt(['Reporter Countries'], years, 'year', 'value')

    # Convert data to time series
    data_long['year'] = data_long['year'].map(
        lambda x: x.lstrip('Y'))  # strip Y from year names for easy converting to ts
    data_long.year = pd.to_datetime(data_long.year)

    # Reshape data from long to wide, turn countries into columns
    df_original = data_long.pivot(index='year', columns='Reporter Countries', values='value')

    # Calculate 3-year rolling mean
    rolled = df_original.rolling(3).mean().add(1)
    df_normalized = rolled[2:]

    # Filter for countries selected by user
    df_original_selected = df_original[countries]
    df_normalized_selected = df_normalized[countries]

    return df_original_selected, df_normalized_selected


def plot_selected(df1, df2, item, element, countries, item_unit):

    # Visually compare original data and normalized data
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(12, 4))
    ax1.plot(df1)
    ax2.plot(df2)

    ax1.set_xlabel('Original data', fontsize=12)
    ax2.set_xlabel('Three-year Rolling Average',fontsize=12)

    ax1.set_ylabel('{}'.format(item_unit[0]), fontsize=12)
    fig.suptitle('Annual {} {} by Countries'.format(item, element), fontsize=12)
    plt.legend(title='Country', labels=countries, loc='upper right', fontsize=10,
               handlelength=3, borderpad=1.2, labelspacing=0.6, bbox_to_anchor=(1.4, 1.05))
    plt.tight_layout(pad =1.5)

    return fig


# Function 4 : plotting
def plot_forecast(country, df_processed, forecast_values, lower_ci_values, upper_ci_values,
                  item,element,unit, n_periods=8):

    '''Function to plot selected country data including 3-year average, predicted and 95% CI'''
    index_future = pd.date_range(start='2018', periods= n_periods, freq='AS-JAN')
    index = pd.date_range(start='1990', end=index_future.tolist()[-1], freq='AS-JAN')

    fig, ax = plt.subplots()
    ax.plot(df_processed[country]['1990-01-01':], label='Three-year average')
    ax.plot(forecast_values[country]['1990-01-01':], color='red', label='Predicted')

    # Plot CI
    ax.fill_between(index_future, lower_ci_values[country],
                    upper_ci_values[country], color='gray', alpha=.5, label='95% confidence interval')
    ax.set_title('Predicted {} {} by {}'.format(item, element, country))
    ax.set_ylabel(unit[0])
    ax.legend(loc='upper left', frameon=False)
    # set y axis to be begin at 0 and max at 1.2 times of maximum value in the plot
    plt.ylim(0, max(max(df_processed[country]), max(forecast_values[country]), max(upper_ci_values[country])) * 1.2)


    return fig
