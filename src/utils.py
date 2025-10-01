# src/utils.py
import json
import numpy as np
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def save_metrics(name, metrics):
    results_path = os.path.join(PROJECT_ROOT, "results", "metrics")
    os.makedirs(results_path, exist_ok=True)

    filepath = os.path.join(results_path, f"{name}.json")
    with open(filepath, "w") as f:
        json.dump(metrics, f)

def load_metrics(name):
    results_path = os.path.join(PROJECT_ROOT, "results", "metrics")
    filepath = os.path.join(results_path, f"{name}.json")
    with open(filepath, "r") as f:
        return json.load(f)

def save_predictions(name, y_pred):
    results_path = os.path.join(PROJECT_ROOT, "results", "predictions")
    os.makedirs(results_path, exist_ok=True)

    filepath = os.path.join(results_path, f"{name}_preds.npy")
    np.save(filepath, y_pred)

def load_predictions(name):
    results_path = os.path.join(PROJECT_ROOT, "results", "predictions")
    filepath = os.path.join(results_path, f"{name}_preds.npy")
    return np.load(filepath)
