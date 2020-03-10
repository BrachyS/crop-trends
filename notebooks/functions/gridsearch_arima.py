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
    
