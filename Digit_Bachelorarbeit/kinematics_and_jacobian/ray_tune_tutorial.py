from ray import tune, train
import numpy as np


def simdigit(kp, kd):
    ...



def objective(config):  # ①
    score = config["a"] ** 2 + config["b"]
    
    error = np.mean(np.square(ist_pos - soll_pos))
    
    return {"score": error}


search_space = {  # ②
    "a": tune.grid_search([0.001, 0.01, 0.1, 1.0]),
    "b": tune.grid_search([1, 2, 3]),
}

tuner = tune.Tuner(objective, param_space=search_space)  # ③

results = tuner.fit()
print(results.get_best_result(metric="score", mode="max").config)