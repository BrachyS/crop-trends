def stationarity(df):
    '''function to run Dickey-Fuller test on time series for each country,
    return list of countries that did not pass the test, and total number of countries'''
    non_stationary = []
    
    for country in df.columns:
        stationarity = adfuller(df[country].dropna())
        
        if stationarity[1] <= 0.05:
            non_stationary.append((country,stationarity[1]))
    non_stationary = pd.DataFrame(non_stationary, columns=['country','p_value'])
    return non_stationary['country'].tolist(), len(non_stationary)
