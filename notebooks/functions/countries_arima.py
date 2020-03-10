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

# Step 2: Grid search for order(p,d,q) of ARIMA model
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
                order = (p,d,q)
                try:
                    mse = eval_arima(data, order)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                    #print('ARIMA%s MSE=%.2E' %(order, mse)) # print out results for each configuration
                except:
                    continue
    print(data.name, best_cfg, 'MSE=%.2E' %best_score)
    return data.name, best_cfg, best_score
    

# Step 3: Loop through all countries to make predictions 
def countries_arima(data):

    results = []
    for country in data.columns:
        country_result = gridsearch_arima(data[country], p_values, d_values, q_values)
        results.append(country_result)
    results = pd.DataFrame(results, columns=['country','best_arima','mse'])
