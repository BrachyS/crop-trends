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

# Customized functions to do ARIMA modeling
import functions.crop_by_country_arima_analyses as ca

# Read in processed data
items_by_country = pd.read_csv('../data/processed/items_by_country.csv',index_col=0)
print(items_by_country.shape)

# Put name for years into a column
# The years list is used by the function below when reshaping data frame
year = items_by_country.columns[5:-1].tolist() # Select year 1986-2017

items_by_country.head()