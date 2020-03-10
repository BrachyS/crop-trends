# ref: https://machinelearningmastery.com/grid-search-arima-hyperparameters-with-python/
# Step 1: Evaluate an ARIMA model use MSE 

def eval_arima(data, order):
    '''function to split a time series into train and test set, build an ARIMA model
      use rolling one-at-a-time predictions, and reture mean squared error of the model'''
    
    # split dataset into training and testing
    train_size = int(len(data)*0.66)
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
