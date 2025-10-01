#!/usr/bin/env python
# coding: utf-8

# # Fourier regression implementation

# 
# ### **Overview of Time Series Analysis**
# 
# Time series analysis involves studying data collected over time to uncover patterns, relationships, and underlying structures. A key first step is visualizing the data over time, which can suggest appropriate models and highlight important features. There are two major—but complementary—approaches to time series analysis: ([1][1])
# 
# #### 1. Time Domain Approach
# 
# * **Focus**: Models the **dependence of current values on past values**.
# * **Method**: Uses **linear regression** techniques to express a variable $ ( x_t ) $ as a function of its own lagged values (e.g., $ ( x_{t-1}, x_{t-2} ) $) and/or lagged values of other variables (e.g., $ ( y_{t-1}, y_{t-2} )$).
# * **Applications**:
# 
#   * **Forecasting** (e.g., economic data like GDP or unemployment)
#   * **Smoothing and filtering**, often using **local regression**, **polynomials**, or **splines**
# * **Advantages**: Often performs better for **shorter time series** and is widely used in economics. ([1][1])
# 
# ---
# 
# #### 2. Frequency Domain Approach
# 
# * **Focus**: Identifies **periodic patterns or cycles** within the data using **sinusoidal components**.
# * **Method**: Applies **spectral analysis** to decompose a time series into its frequency components, producing a **power spectrum** that shows how variance is distributed across frequencies.
# * **Concepts**:
# 
#   * **Frequency (ω)** is measured in cycles per time point.
#   * **Period** is the number of time points per cycle.
#   * **Coherence** measures the strength of the relationship between two series at specific frequencies.
# * **Applications**: Useful in fields such as **engineering**, **biomedicine**, **oceanography**, and **geophysics**, where natural or environmental cycles are significant. ([1][1])
# 
# ---
# 
# #### Unifying Theme: The Linear Model
# 
# * Linear models are **central** to both approaches:
# 
#   * In the **time domain**, regression is used to model future values based on past observations.
#   * In the **frequency domain**, regression with **sine and cosine inputs** leads to the **periodogram** and **spectral analysis**.
# * **Time-invariant linear filters** play a key role, functioning similarly to regression models and enabling both forecasting and smoothing.
# * The use of **least squares estimation** for model parameters connects time series analysis directly to classical statistical techniques. ([1][1])
# 
# 
# [1]: https://sistemas.fciencias.unam.mx/~ediaz/Cursos/Estadistica3/Libros/Time%20Series%20Analysis%20and%20Its%20Applications.pdf "Time Series Analysis and Its Applications (2ed) | Shumway, R. H., & Stoffer, D. S."

# ### Cyclical Behavior and Periodicity
# Time series data often contain multiple coexisting frequencies depending on the context.
# 
# Frequencies $ (\omega) $ are measured in cycles per time point, and understanding their significance depends on the context. An important related concept is the period $(T)$, defined as the number of time points required to complete one full cycle.
# 
# $$ T = \frac{1}{\omega} $$
# 
# In order to define the rate at which a series oscillates, we first define a cycle as one complete period of a sine or cosine function defined over a time interval of length $ 2\pi $.

# ### Fourier series
# 
# In its basic form, the model can be written as:
# 
# $$
# y(t) = a_0 + \sum_{k=1}^{N} \left[ a_k \cos\left( \frac{2\pi k t}{T} \right) + b_k \sin\left( \frac{2\pi k t}{T} \right) \right] + \varepsilon
# $$
# 
# Where:
# 
# * $t$ is the independent variable (often time),
# * $T$ is the period of the signal,
# * $N$ is the number of harmonics (or degrees),
# * $a_k$, $b_k$ are Fourier coefficients,
# * $\varepsilon$ is the noise.
# 

# In[12]:


import numpy as np

# Sines and cosines in a frecuency domain
def fourier_features(x, degree, period):
    x = np.asarray(x).reshape(-1, 1)
    features = [np.ones_like(x)]
    for i in range(1, degree + 1):
        features.append(np.sin(2 * np.pi * i * x / period))
        features.append(np.cos(2 * np.pi * i * x / period))
    return np.hstack(features)


# In[13]:


from scipy.fft import rfft, rfftfreq

def estimate_period(x, y):
    N = len(x)
    T = x[1] - x[0] # Sampling interval (assume uniform sampling)
    yf = rfft(y) # real fourier transform
    xf = rfftfreq(N, T) # Corresponding frequencies
    freq = xf[np.argmax(np.abs(yf[1:])) + 1]  # Dominant frequency (avoids peak at zero)
    
    if freq == 0:
        raise ValueError("No dominant frequency found.")
    
    return 1 / freq


# In[14]:


# Fourier features since sine and cosine have period 2π
# Input x is transformed, so that the minimum value corresponds to 0 and the maximum to 2π

def normalize_x(x):
    return 2 * np.pi * (x - x.min()) / (x.max() - x.min())


# In[ ]:


import sys
import os

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from utils import save_metrics, save_predictions

sys.path.append(os.path.abspath(".."))

def fit_fourier_models(x_train, Y_train, x_test, y_test, degree_rang = range(2,15)):
    
    num_models = Y_train.shape[1]
    Y_pred = np.zeros((len(x_test), num_models))
    MSE_train = np.zeros(num_models)
    MSE_test = np.zeros(num_models)
    bias = np.zeros(num_models)

    for i in range(num_models):
        y_i = Y_train[:, i]
        best_mse_test = np.inf
        best_pred = None
        best_mse_train = None

        # Estimate the period
        period = estimate_period(x_train, y_i)

        for degree in degree_rang:
            # Normalize x inputs
            x_train_norm = normalize_x(x_train)
            x_test_norm = normalize_x(x_test)

            # Calculate fourier features
            X_train = fourier_features(x_train_norm, degree, period)
            X_test = fourier_features(x_test_norm, degree, period)

            # Create model and fit using linear regression
            model = LinearRegression()
            model.fit(X_train, y_i)

            # Make predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

            # Compute MSE for training and test
            mse_train = mean_squared_error(y_i, y_pred_train)
            mse_test = mean_squared_error(y_test, y_pred_test)

            if mse_test < best_mse_test:
                best_mse_test = mse_test
                best_pred = y_pred_test
                best_mse_train = mse_train

        # Save best prediction and errors
        Y_pred[:, i] = best_pred
        MSE_train[i] = best_mse_train
        MSE_test[i] = best_mse_test
        bias[i] = np.mean(np.abs(best_pred - y_test)) 

    # Final metrics
    Bias = np.mean(bias)
    Variance = np.var(Y_pred, axis=1).mean()  # Mean of variances across all test points

    roundN = 8
    metrics = {
        "Name": "Fourier",
        "MSE_train": round(float(np.mean(MSE_train)), roundN),
        "MSE_test": round(float(np.mean(MSE_test)), roundN),
        "Bias": round(float(Bias), roundN),
        "Variance": round(float(Variance), roundN),
    }

    save_metrics("fourier", metrics)
    save_predictions("fourier", Y_pred)


# ## References
# [1] R. H. Shumway and D. S. Stoffer, *Time Series Analysis and Its Applications: With R Examples*, 2nd ed. Cham, Switzerland: Springer, 2005.
# 
