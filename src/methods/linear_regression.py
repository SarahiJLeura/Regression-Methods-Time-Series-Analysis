#!/usr/bin/env python
# coding: utf-8

# # Linear regression implementation

# **Definition & Purpose**
# 
# * Linear Regression is a **supervised learning** algorithm used to model the relationship between one or more independent variables (inputs) and a dependent variable (output) by fitting a linear equation. ([GeeksforGeeks][1])
# * It assumes the output changes at a constant rate with respect to each input. ([GeeksforGeeks][1])
# 
# <a href="https://www.geeksforgeeks.org/machine-learning/ml-linear-regression/" target="_blank">
#     <figure>
#       <img src="https://media.geeksforgeeks.org/wp-content/uploads/20231129130431/11111111.png" alt="GeeksforGeeks" width="800">
#       <figcaption>Linear regression. GeeksforGeeks.</figcaption>
#     </figure>
# </a> 
# 
# **Equation / Hypothesis Function**
# 
# * For **simple linear regression** (one feature):
#   $$
#   \hat{y} = \theta_0 + \theta_1 x
#   $$
# * For **multiple linear regression** (multiple features):
#   $$
#   \hat{y} = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \cdots + \theta_n x_n
#   $$
#   ([GeeksforGeeks][1])
# 
# **Finding the Best Fit (Optimization)**
# 
# * Use the **Least Squares Method** to minimize the sum of squared residuals:
#   $$
#   \sum (y_i - \hat y_i)^2
#   $$
#   This yields optimal parameter values (βs). ([GeeksforGeeks][1])
# * **Gradient Descent** is an iterative optimization method often used to update parameters (βs) to reduce the loss. ([GeeksforGeeks][1])
# 
# **Cost / Loss Function**
# 
# * **Mean Squared Error (MSE)** is commonly used:
#   $$
#   \text{MSE} = \frac{1}{n} \sum (y_i - \hat y_i)^2
#   $$
#   Lower MSE → better fit. ([GeeksforGeeks][1])
# 
# **Assumptions of Linear Regression**
# 
# 1. Linearity: The relationship between inputs and output is linear. ([GeeksforGeeks][1])
# 2. Independence of errors (residuals). ([GeeksforGeeks][1])
# 3. Homoscedasticity: Constant variance of errors across all levels of input. ([GeeksforGeeks][1])
# 4. Normality of errors (residuals). ([GeeksforGeeks][1])
# 5. No multicollinearity (for multiple regression): inputs not highly correlated. ([GeeksforGeeks][1])
# 6. No autocorrelation (especially for time‑series data). ([GeeksforGeeks][1])
# 
# **Evaluation Metrics**
# 
# * **Mean Squared Error (MSE)**
# * **Mean Absolute Error (MAE)**: average absolute difference between predictions and actuals ([GeeksforGeeks][1])
# * **Root Mean Squared Error (RMSE)**: square root of the MSE ([GeeksforGeeks][1])
# * **Coefficient of Determination (R²)**: proportion of variance in the dependent variable explained by the model (value between 0 and 1) ([GeeksforGeeks][1])
# * **Adjusted R²**: adjusts R² by penalizing unnecessary predictors (useful in multiple regression) ([GeeksforGeeks][1])
# 
# **Regularization Techniques** *(to avoid overfitting / handle multicollinearity)*
# 
# * **Ridge Regression (L2 regularization)**: adds penalty proportional to squared magnitude of coefficients ([GeeksforGeeks][1])
# * **Lasso Regression (L1 regularization)**: adds penalty proportional to absolute values of coefficients (can drive some coefficients to zero) ([GeeksforGeeks][1])
# * **Elastic Net**: combines L1 and L2 penalties ([GeeksforGeeks][1])
# 
# **Advantages**
# 
# * Simple to understand, interpret, and implement ([GeeksforGeeks][1])
# * Computationally efficient ([GeeksforGeeks][1])
# * Provides insight into the relationship between variables (coefficients have meaning) ([GeeksforGeeks][1])
# * Serves as a baseline for comparing more complex models ([GeeksforGeeks][1])
# 
# **Disadvantages / Limitations**
# 
# * Assumes linear relationships (so it performs poorly if the true relation is nonlinear) ([GeeksforGeeks][1])
# * Sensitive to outliers — large deviations can disproportionately influence the model ([GeeksforGeeks][1])
# * Multicollinearity among features degrades coefficient stability ([GeeksforGeeks][1])
# * May underfit when the relationship is more complex than a linear one ([GeeksforGeeks][1])
# 
# **Applications**
# 
# * Predicting real estate prices based on features (size, location, etc.) ([GeeksforGeeks][1])
# * Forecasting economic indicators, stock prices, etc. ([GeeksforGeeks][1])
# * Analyzing relationships in healthcare, marketing, etc.
# 
# [1]: https://www.geeksforgeeks.org/machine-learning/ml-linear-regression/ "Linear Regression in Machine learning - GeeksforGeeks"
# 

# ### Implementation code

# In[ ]:


import sys
import os
sys.path.append(os.path.abspath(".."))

import numpy as np
from sklearn.metrics import mean_squared_error
from utils import save_metrics, save_predictions

def fit_linear_models(x_train, Y_train, x_test, y_test):
    num_realizations = Y_train.shape[1]

    # Initialize result storage
    Y_pred = np.zeros((len(x_test), num_realizations))
    MSE_train = np.zeros(num_realizations)
    MSE_test = np.zeros(num_realizations)
    bias_per_model = np.zeros((len(x_test), num_realizations)) 

    # Loop over each realization
    for i in range(num_realizations):
        y_i = Y_train[:, i]  # Current noisy realization

        #Fit line: y = a0*x + a1
        a0, a1 = np.polyfit(x_train, y_i, 1)

        # Make predictions
        y_train_pred = a0 * x_train + a1
        y_test_pred = a0 * x_test + a1

        # Compute MSE for training and test
        mse_tr = mean_squared_error(y_i, y_train_pred)
        mse_te = mean_squared_error(y_test, y_test_pred)

        Y_pred[:, i] = y_test_pred
        MSE_train[i] = mse_tr
        MSE_test[i] = mse_te
        bias_per_model[:,i] = np.abs(y_test_pred - y_test)

    Bias = np.mean(bias_per_model)

    Variance = np.var(Y_pred, axis=1).mean()  # Mean of variances across all test points

    roundN = 8
    metrics = { # Create dictionary with metrics
        "Name": "Linear",
        "MSE_train": round(float(np.mean(MSE_train)), roundN),
        "MSE_test": round(float(np.mean(MSE_test)), roundN),
        "Bias": round(float(Bias), roundN),
        "Variance": round(float(Variance), roundN),
    }
    
    save_metrics("linear", metrics)
    save_predictions("linear", Y_pred)

