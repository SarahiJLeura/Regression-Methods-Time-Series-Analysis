#!/usr/bin/env python
# coding: utf-8

# # Kernel regression implementation

# *Based on: Cuevas-Tello, Juan Carlos. (2020). Handouts on Regression Algorithms.  
# DOI: [10.13140/RG.2.2.20046.74562](https://doi.org/10.13140/RG.2.2.20046.74562)*
# 
# ---
# 
# ## Kernel Methods
# - **Definition:**  
#   A kernel is a two-variable function $(K(t', t))$ that maps data into a higher-dimensional *feature space* through a transformation:
#   $$
#   K(t', t) = \langle \phi(t'), \phi(t) \rangle
#   $$
#   This is known as the **kernel trick**.
# 
# - **Purpose:**  
#   Allows nonlinear regression problems to be treated as linear in the transformed feature space (Reproducing Kernel Hilbert Space, RKHS).
# 
# - **Common Kernels:**
#   - Polynomial kernel
#   - Gaussian (RBF) kernel:  
#     $$
#     K(t', t) = \exp\left(-\frac{|t - t'|^2}{\omega^2}\right)
#     $$
#   - Sigmoid kernel
# 
# - **Regression model (Representer Theorem):**
#   $$
#   f(t) = \sum_{j=1}^n \alpha_j K(t_j, t)
#   $$
#   where $(\alpha_j)$ are weights learned from data.
# 
# - **Key Points:**
#   - Requires computing a **Gram matrix** of pairwise similarities.  
#   - Learning involves solving linear systems (often via pseudo-inverse or SVD).  
#   - **Parameter to tune:** kernel width/spread $(\omega)$.  
#   - **Advantage:** flexible, good for nonlinear data.  
#   - **Challenge:** choice of $(\omega)$ is critical; too small → overfitting, too large → oversmoothing. Cross-validation is recommended.

# ### Implementation code

# In[ ]:


import sys
import os
sys.path.append(os.path.abspath(".."))

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
from utils import save_metrics, save_predictions

def fit_kernel_ridge_models(x_train, y_train, x_test, y_test,
                           alpha_range = np.logspace(-6, 1, 20),
                           gamma_range = np.logspace(-3, 2, 20)):
    
    n_realizations = y_train.shape[1]
    Y_pred = np.zeros((len(x_test), n_realizations))
    MSE_train = np.zeros(n_realizations)
    MSE_test = np.zeros(n_realizations)
    biases = np.zeros((len(x_test), n_realizations))

    for i in range(n_realizations):
        y_train_i = y_train[:, i]
        best_mse_test = np.inf
        best_pred = None
        # Test different alphas and gammas to select the best model
        for alpha in alpha_range:
            for gamma in gamma_range:
                scaler = StandardScaler()
                # Scale the input x
                X_train_scaled = scaler.fit_transform(x_train.reshape(-1, 1))
                X_test_scaled = scaler.transform(x_test.reshape(-1, 1))
                # Create model
                model = KernelRidge(alpha=alpha, kernel='rbf', gamma=gamma)
                # Fit model
                model.fit(X_train_scaled, y_train_i)
                # Make predictions
                y_pred_train = model.predict(X_train_scaled)
                y_pred_test = model.predict(X_test_scaled)
                # Compute MSE for training and test
                mse_train = mean_squared_error(y_train_i, y_pred_train)
                mse_test = mean_squared_error(y_test, y_pred_test)
                # Find the best model for each realization
                if mse_test < best_mse_test:
                    best_mse_test = mse_test
                    best_pred = y_pred_test
                    best_mse_train = mse_train
        # Save the values, then an average will be made with them
        Y_pred[:, i] = best_pred
        MSE_train[i] = best_mse_train
        MSE_test[i] = best_mse_test
        # Calculate bias depending of the best prediction and y_test
        biases[:, i] = np.abs(best_pred - y_test)

    Bias = np.mean(biases)
    Variance = np.var(Y_pred, axis=1).mean()
    
    roundN= 8    
    metrics = { # Create dictionary with metrics
        "Name": "Kernel Ridge",
        "MSE_train": round(float(np.mean(MSE_train)), roundN),
        "MSE_test": round(float(np.mean(MSE_test)), roundN),
        "Bias": round(float(Bias), roundN),
        "Variance": round(float(Variance), roundN),
    }

    save_metrics("kernel", metrics)
    save_predictions("kernel", Y_pred)

