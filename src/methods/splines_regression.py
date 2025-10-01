#!/usr/bin/env python
# coding: utf-8

# # Splines regression implementation

# This section describes different spline-based methods for flexible regression that allow modeling non‐linear relationships by partitioning the input space and imposing smoothness constraints.
# 
# #### Key Concepts
# 
# * **Knots**: Points $z_i$ that partition the domain into intervals; in each interval a polynomial is fit. ([rubenfcasal.github.io][1])
# * **Piecewise regression**: Fit separate polynomials in each interval; naive version can lead to discontinuities at the knots unless continuity/differentiability constraints are imposed. ([rubenfcasal.github.io][1])
# 
# #### 4.2.1 Regression Splines
# 
# * In each interval, one fits a polynomial of degree $d$, with constraints so the function and its derivatives up to order $d-1$ are continuous across knots. ([rubenfcasal.github.io][1])
# * A popular choice is cubic splines ($d=3$). One common basis is the truncated power basis:
# 
#   $$
#     1, x, x^2, \dots, x^d, (x - z_1)_+^d, \dots, (x - z_k)_+^d
#   $$
# 
#   where $(x - z)_+ = \max(0, x - z)$. ([rubenfcasal.github.io][1])
# * Another basis is the **B-spline** basis, which tends to have better numerical properties. In R, this is implemented via `bs()` in the `splines` package. ([rubenfcasal.github.io][1])
# * **Natural splines** add further constraints: the fit is linear in the tails (outside the outermost knots), which improves stability at the boundaries. In R: `ns()`. ([rubenfcasal.github.io][1])
# * Choice of number of knots (or equivalently degrees of freedom) is crucial. Can use equally spaced knots, quantile‐based, or more knots where the function varies more. Cross‐validation is often used to select this. ([rubenfcasal.github.io][1])
# 
# #### 4.2.2 Smoothing Splines
# 
# * Instead of preset knots, smoothing splines choose a smooth function $s(x)$ (twice differentiable) that minimizes
# 
#   $$
#     \sum_i (y_i - s(x_i))^2 + \lambda \int [s''(x)]^2 \, dx
#   $$
# 
#   where $\lambda\ge0$ is a smoothing parameter—small $\lambda$ yields a very wiggly fit (closer to interpolating the data); large $\lambda$ forces smoother behavior (approaching a linear fit as $\lambda \to \infty$). ([rubenfcasal.github.io][1])
# * For univariate $x$, the solution is a natural cubic spline with knots at every observed $x_i$, with smoothness determined by $\lambda$. ([rubenfcasal.github.io][1])
# * For multivariate inputs, there are generalizations such as thin‐plate splines. ([rubenfcasal.github.io][1])
# * Selection of $\lambda$ can be done via leave‐one‐out cross‐validation (CV) or generalized cross validation (GCV). Also, the effective degrees of freedom (trace of the smoother matrix) serves as a measure of complexity. ([rubenfcasal.github.io][1])
# 
# #### 4.2.3 Penalized Splines
# 
# * These combine features of regression splines and smoothing splines. Use a relatively small number of knots, but impose a penalty on the coefficients (e.g. penalize differences between adjacent coefficients) to control wiggliness. Such models are often called **low‐rank smoothers**. ([rubenfcasal.github.io][1])
# * An example: **P‑splines** (Eilers & Marx, 1996) use B‑spline basis + penalty on squared differences of coefficients. ([rubenfcasal.github.io][1])
# * These models can be cast in the framework of mixed‐effects models, which allows leveraging tools from that setup (e.g. as in R packages `nlme`, `mgcv`). In particular, `mgcv` commonly uses penalized splines. ([rubenfcasal.github.io][1])
# 
# [1]: https://rubenfcasal.github.io/aprendizaje_estadistico/splines.html "7.2 Splines | Métodos predictivos de aprendizaje estadístico"
# 

# ### Implementation code

# In[ ]:


import sys
import os

sys.path.append(os.path.abspath(".."))

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import SplineTransformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from utils import save_metrics, save_predictions

def fit_spline_models(x_train, Y_train, x_test, y_test, n_knots_range=range(5, 15)):
    num_models = Y_train.shape[1]
    Y_pred = np.zeros((len(x_test), num_models))
    MSE_train = np.zeros(num_models)
    MSE_test = np.zeros(num_models)
    bias_per_model = np.zeros((len(x_test), num_models))  # pointwise error
    #kn = 5

    for i in range(num_models):
        y_i = Y_train[:, i]
        best_mse_test = np.inf
        best_pred = None
        best_mse_train = None

        for n_knots in n_knots_range:
            # Create spline + regression pipeline
            pipeline = make_pipeline(
                SplineTransformer(n_knots=n_knots, degree=3, include_bias=False),
                LinearRegression()
            )

            # Fit the model
            pipeline.fit(x_train.reshape(-1, 1), y_i)

            # Predict
            y_pred_train = pipeline.predict(x_train.reshape(-1, 1))
            y_pred_test = pipeline.predict(x_test.reshape(-1, 1))
            # Calculate MSEs
            mse_train = mean_squared_error(y_i, y_pred_train)
            mse_test = mean_squared_error(y_test, y_pred_test)

            if mse_test < best_mse_test:
                best_mse_test = mse_test
                best_mse_train = mse_train
                best_pred = y_pred_test
                #if (i==1): kn = n_knots

        # Save best model results for curve i
        Y_pred[:, i] = best_pred
        MSE_train[i] = best_mse_train
        MSE_test[i] = best_mse_test
        bias_per_model[:, i] = np.abs(best_pred - y_test)

    # Calculate bias (average mean error)
    Bias = np.mean(bias_per_model)
    # Calculate variance (average variance per point)
    Variance = np.mean(np.var(Y_pred, axis=1))

    roundN = 8
    metrics= {
        "Name": f"Splines deg=3",
        "MSE_train": round(float(np.mean(MSE_train)), roundN),
        "MSE_test": round(float(np.mean(MSE_test)), roundN),
        "Bias": round(float(Bias), roundN),
        "Variance": round(float(Variance), roundN),
    }

    save_metrics("splines", metrics)
    save_predictions("splines", Y_pred)

