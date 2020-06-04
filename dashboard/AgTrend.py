import streamlit as st
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
import time
from functions import dash_functions as ca # Import Customized ARIMA functions

'''## **AgTrend**'''
'''## Global Agricultural Commodities Production and Trade Forecast (2020-2025)'''
'''(Check back for updates!)'''
''' Author: Shuangshuang (Sabrina) Liu '''

data_path = "../data/processed/items_by_country.csv"

with st.spinner('Loading data...'):
    @st.cache
    def load_data():
        data = pd.read_csv(data_path, index_col=0)
        return data

    items_by_country = load_data()

with st.spinner('Data loaded.'):
    st.write('')

years = items_by_country.columns[5:-1].tolist()  # Select year 1986-2017

####### Sidebars
'#### Step 1:  Select an item and countries from left sidebar to visualize data trend!'
# Select element
element = items_by_country.Element.unique().tolist()
option_element = st.sidebar.selectbox('Select an category', element)

#Select an item
item_list = items_by_country.loc[items_by_country['Element'] == option_element,:].Item.unique().tolist()
option_item = st.sidebar.selectbox('Select an item to inspect',item_list)

option_unit = items_by_country.loc[(items_by_country['Element'] == option_element)
                                   & (items_by_country['Item'] == option_item),'Unit'].unique().tolist()

#Select country
country_list = items_by_country.loc[(items_by_country['Element']== option_element)
                                    &(items_by_country['Item']== option_item),
                                    'Reporter Countries'].unique().tolist()

#### Filter out countries with data eligible for modeling ####
eligible_original, eligible_processed = ca.crop_country_preprocess(data=items_by_country,
                                                years=years,item=option_item, element=option_element, plot=False)

#Select elegible country
eligible_list = eligible_processed.columns.tolist()

option_country = st.sidebar.multiselect('Explore from {} countries that have data eligible for modeling '.format(
                                        len(eligible_list),option_element,option_item), eligible_list)

######## Plotting

data_original, data_normalized = ca.country_selected(df=items_by_country, years=years,
                                                    item=option_item, element=option_element, countries=option_country)
# Plot item-by-country data
if data_original.shape[1] == 0:
    st.write('No country selected')
else:
    figure = ca.plot_selected(df1=data_original, df2=data_normalized, item=option_item,
                          element=option_element, countries=option_country, item_unit=option_unit)
    figure

# Option to display data
if st.sidebar.checkbox('Show selected raw data'):
    st.subheader('{} for {} ({}) (1986-2017)'.format(option_element,option_item, option_unit[0]))
    st.write(data_original)

####################
'''#### Step 2:  Fit an ARIMA model for the item-by-country combination (check box from left sidebar)'''
country = st.sidebar.selectbox('Select one country for modeling ',eligible_list)

if st.sidebar.checkbox('Fit ARIMA models for selected country (scroll down for forecast plot)'):
    @st.cache
    def model(df):
        if df.shape[1] == 0:
            st.write('No country selected')
        else:
            df_arima = ca.crop_country_arima(df, item=option_item, element=option_element, plot=False)
            forecast, lower_ci, upper_ci = ca.crop_country_forecast(df, df_arima)
            return df_arima,forecast, lower_ci, upper_ci

    results = model(eligible_processed[[country]])


    with st.spinner('Modeling completed.'):
        time.sleep(1)
    st.success('Done.')
    st.write(results[0])

    '''#### Step 3:  Plot forecast of eligible countries (with sufficient data) into 2025'''
    ffig = ca.plot_forecast(country, eligible_processed
                            , results[1], results[2], results[3],
                            option_item, option_element, option_unit)
    ffig

    '''Notes:'''
    '''best_arima: optimized parameters for ARIMA models. Ref: https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average
    (currently, only stationary time series were included in modeling so the second parameter (d) is always 0)'''
    '''mse: Mean squared error'''
    '''mspe: Mean Absolute Percentage Error'''
    '''corr: Correlation between predictions and test data'''
    '''minmax: Min-Max Error'''
    ''' Data source: http://www.fao.org/faostat/en/#data/TM '''
    ''' Please note: Forecasting results are not reviewed by domain experts and are for references only.'''


