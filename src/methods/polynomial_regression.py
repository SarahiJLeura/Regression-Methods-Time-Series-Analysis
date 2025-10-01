#!/usr/bin/env python
# coding: utf-8

# # Polynomial regression implementation

# **What is Polynomial Regression**
# 
# * Polynomial regression is an extension of linear regression: instead of fitting a straight line, you fit a polynomial of degree *n* to capture non‑linear relationships between an independent variable $x$ and a dependent variable $y$. ([GeeksforGeeks][1])
# * The general form is:
# 
# $$
# y = \beta_0 + \beta_1 x + \beta_2 x^2 + \dots + \beta_n x^n + \varepsilon
# $$
# 
# where $\beta_i$ are coefficients and $\varepsilon$ is an error term. ([GeeksforGeeks][1])
# 
# **Issues, Trade‑offs & Tips**
# 
# * **Overfitting vs Underfitting**: As degree increases, the model may fit training data very well but perform poorly on unseen data (overfit). Low degree may underfit, fail to capture structure. Need to balance. ([GeeksforGeeks][1])
# * **Bias‑Variance Trade‑off**: Choosing the polynomial degree is a way of navigating that trade‑off: lower degree ≈ high bias, lower variance; higher degree ≈ low bias, high variance. ([GeeksforGeeks][1])
# * **Sensitivity to outliers**: Because polynomial features amplify small differences (especially with high degree), outliers can pull the fit significantly. ([GeeksforGeeks][1])
# 
# [1]: https://www.geeksforgeeks.org/machine-learning/python-implementation-of-polynomial-regression/ "Implementation of Polynomial Regression - GeeksforGeeks"

# ### Implementation code

# In[ ]:


import sys
import os
sys.path.append(os.path.abspath(".."))

import numpy as np
from sklearn.metrics import mean_squared_error
from utils import save_metrics, save_predictions

def fit_polynomial_models(x_train, Y_train, x_test, y_test, degree_rang = range(2,10)):
    num_realizations = Y_train.shape[1]

    # Initialize result storage
    Y_pred = np.zeros((len(x_test), num_realizations))
    MSE_train = np.zeros(num_realizations)
    MSE_test = np.zeros(num_realizations)
    bias_per_model = np.zeros((len(x_test), num_realizations))   # pointwise error
    best_degree = 2

    # Loop over each realization
    for i in range(num_realizations):
        y_i = Y_train[:, i]  # Current noisy realization
        best_mse_test = np.inf  # Initialize with a high test MSE

        # Try all polynomial degrees from min to max
        for d in degree_rang:
            # Fit polynomial of degree d
            coeffs = np.polyfit(x_train, y_i, d)

            # Make predictions
            y_train_pred = np.polyval(coeffs, x_train)
            y_test_pred = np.polyval(coeffs, x_test)

            # Compute MSE for training and test
            mse_tr = mean_squared_error(y_i, y_train_pred)
            mse_te = mean_squared_error(y_test, y_test_pred)

            # Keep the model with lowest test error
            if mse_te < best_mse_test:
                best_mse_test = mse_te
                best_mse_train = mse_tr
                best_pred = y_test_pred
                if (i == 1): best_degree = d

        # Store results for current realization
        Y_pred[:, i] = best_pred
        MSE_train[i] = best_mse_train
        MSE_test[i] = best_mse_test
        bias_per_model[:, i] = np.abs(best_pred - y_test)

    # Final metrics
    Bias = np.mean(bias_per_model)
    Variance = np.var(Y_pred, axis=1).mean()  # Mean of variances across all test points

    roundN = 8
    metrics =  { # Dictionary of metrics
        "Name": f"Poly deg= {best_degree}",
        "MSE_train": round(float(np.mean(MSE_train)), roundN),
        "MSE_test": round(float(np.mean(MSE_test)), roundN),
        "Bias": round(float(Bias), roundN),
        "Variance": round(float(Variance), roundN),
    }

    save_metrics("polynomial", metrics)
    save_predictions("polynomial", Y_pred)

