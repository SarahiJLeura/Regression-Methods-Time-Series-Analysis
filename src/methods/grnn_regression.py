#!/usr/bin/env python
# coding: utf-8

# ## General Regression Neural Networks (GRNN)

# *Based on: Cuevas-Tello, Juan Carlos. (2020). Handouts on Regression Algorithms.  
# DOI: [10.13140/RG.2.2.20046.74562](https://doi.org/10.13140/RG.2.2.20046.74562)*
# 
# ---
# - **Definition:**  
#   A type of neural network for regression introduced by Specht (1991).  
#   Architecture includes:
#   1. **Input Layer**: passes data features.
#   2. **Pattern Layer**: applies Gaussian basis functions  
#      $(\phi_i = \exp\left(-\frac{\|X - x_i\|^2}{2\sigma^2}\right))$.
#   3. **Summation Layer**:  
#      - $(S_s = \sum_i \phi_i)$  
#      - $(S_w = \sum_i w_i \phi_i)$  
#      - Output:  
#        $$
#        y = \frac{S_w}{S_s}
#        $$
# 
# - **Learning:**
#   - Weights $(w_i)$ are obtained directly from training outputs (no gradient descent).  
#   - Deterministic — no random initialization.
# 
# - **Key Parameter:**  
#   The **spread** ($(\sigma)$), controls smoothness (similar to $(\omega)$ in kernels).  
# 
# - **Comparison to Kernel Methods:**
#   - Both use Gaussian functions.  
#   - GRNN has only **one parameter** ($(\sigma)$) to tune.  
#   - Kernel methods need solving for weights $(\alpha)$ via linear algebra.  
#   - GRNN is computationally lighter for training.

# In[3]:


import numpy as np

class GRNN:
    def __init__(self, sigma= 0.5):
        self.sigma= sigma

    def fit(self, X, y):
        self.X_train= np.array(X)
        self.y_train= np.array(y)

    def gaussian_kernel(self, x, c):
        return np.exp(-((c - x) ** 2) / (2 * self.sigma ** 2))

    def predict(self, X):
        X = np.ravel(X)  # Ensure 1D
        y_pred = np.zeros_like(X)
        for i, x0 in enumerate(X):
            weights = self.gaussian_kernel(x0, self.X_train)
            y_pred[i] = np.sum(weights * self.y_train) / np.sum(weights)
        return y_pred


# In[6]:

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.metrics import mean_squared_error
from utils import save_metrics, save_predictions


def fit_grnn_models(x_train, Y_train, x_test, y_test, sigma_range = np.linspace(1, 50, 30)):
    n_realizaciones = Y_train.shape[1]
    Y_pred = np.zeros((len(x_test), n_realizaciones))
    MSE_train = np.zeros(n_realizaciones)
    MSE_test = np.zeros(n_realizaciones)
    biases = np.zeros((len(x_test), n_realizaciones))
    
    for i in range(n_realizaciones):
        y_train_i = Y_train[:, i]
        best_mse_test = np.inf
        best_pred = None
        # Test different sigmas to select the best model
        for sigma in sigma_range:
            # Create and fit the model
            grnn = GRNN(sigma)
            grnn.fit(x_train, y_train_i)
            # Make predictions
            y_pred_test = grnn.predict(x_test)
            y_pred_train = grnn.predict(x_train)

            # Compute MSE for training and test
            mse_test = mean_squared_error(y_test, y_pred_test)
            mse_train = mean_squared_error(y_train_i, y_pred_train)
            # Find the best model for each realization
            if mse_test < best_mse_test:
                best_mse_test = mse_test
                best_pred = y_pred_test
                best_mse_train = mse_train
        # Save the values, then an average will be made with them
        Y_pred[:, i] = best_pred
        MSE_train[i] = best_mse_train
        MSE_test[i] = best_mse_test
        biases[:, i] = np.abs(best_pred - y_test)

    Bias = np.mean(biases)
    Variance = np.var(biases)
    
    roundN= 8
    metrics = {
        "Name": "GRNN",
        "MSE_train": round(float(np.mean(MSE_train)), roundN),
        "MSE_test": round(float(np.mean(MSE_test)), roundN),
        "Bias": round(float(Bias), roundN),
        "Variance": round(float(Variance), roundN),
    }

    save_metrics("grnn", metrics)
    save_predictions("grnn", Y_pred)

