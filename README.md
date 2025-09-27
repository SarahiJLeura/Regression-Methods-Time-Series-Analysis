# Regression Methods for Time Series Analysis
This repository provides a brief introduction and Python implementations of various **regression techniques** applied to **noisy and nonlinear time series data**. The main objective is to evaluate the performance of different regression models under varying noise levels using synthetic datasets, particularly in the context of astronomical signal analysis.

---
## Methods Implemented

- **Linear Regression**
- **Polynomial Regression**
- **Fourier Regression**
- **Splines**
- **Kernel Regression**
- **GRNN (General Regression Neural Network)**

Each method is applied to reconstruct nonlinear curves from artificially generated datasets.

---
## Activity Overview

The analysis involves the following steps:

- Use two nonlinear datasets:
  - `DS-5-1-GAP-1-1-N-1_v2.dat` (noise ≈ 0.106%)
  - `DS-5-1-GAP-5-1-N-3_v2.dat` (noise ≈ 0.466%)
- Each dataset contains 201 rows (time samples) and over 100 columns (A1, A2, ..., A100)
- Use the first 100 curves (A1 to A100) per dataset
- For each curve:
  - Train a regression model
  - Measure **Mean Squared Error (MSE)** on:
    - Training data
    - Testing data (ground truth = `DS-5-1-GAP-0-1-N-0_v2.dat`)
  - Measure **bias**:
    - Compute the difference between the mean regression curve and the ground truth
    - Take the mean across all time steps
  - Measure **variance**:
    - Compute the standard deviation of all 100 models at each time step
    - Take the mean across all time steps

---
## Repository Structure
Regression-Methods-Time-Series-Analysis/

│

├── ArtificialData/

│ ├── DS-5-1-GAP-0-1-N-0_v2.dat # Ground truth data (noise = 0)

│ ├── DS-5-1-GAP-1-1-N-1_v2.dat # Low noise level (0.106%)

│ └── DS-5-1-GAP-5-1-N-3_v2.dat # Higher noise level (0.466%)

│

├── src/ # Python scripts for regression methods

│

├── results/ # Output: MSE, bias, variance metrics

│

└── README.md

---
## Goal

This project serves as a benchmarking platform to understand how different regression methods perform in reconstructing underlying patterns from noisy, irregular datasets — a common challenge in fields like **astrophysics**, **signal processing**, and **time series analysis**.

---

## Context and Reference

The regression techniques are inspired by the work presented in:

> Juan C. Cuevas-Tello, Peter Tio, Somak Raychaudhury, Xin Yao, and Markus Harva.  
> *Uncovering delayed patterns in noisy and irregularly sampled time series: An astronomy application.*  
> Pattern Recognition, 43(3):1165 – 1179, 2010.

This study explores time delay estimation in gravitationally lensed quasar signals using kernel-based methods and evolutionary optimization.

**Dataset source:**  
[http://turing.ing.uaslp.mx/~jcctello/time-delay/DS-5/](http://turing.ing.uaslp.mx/~jcctello/time-delay/DS-5/)

---

## Applications
- Astronomical signal reconstruction
- Time delay estimation in astrophysics
- General purpose time series denoising and modeling
- Benchmarking regression models under noise

---

## Getting Started
> You can run each method separately using Python 3.8+ and libraries like NumPy, SciPy, scikit-learn, etc.  
> Each model outputs error metrics and reconstructed curves for further analysis.

---

## Author
**Patricia Sarahi Jimenez-Leura**  
Last updated: XXX 2025

---

## License
This project is open for academic and research purposes. Please cite the original paper if used in publications.

