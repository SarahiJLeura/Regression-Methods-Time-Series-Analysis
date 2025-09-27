# src/utils.py
import json
import numpy as np

def save_metrics(name, metrics):
    with open(f"results/metrics/{name}.json", "w") as f:
        json.dump(metrics, f)

def load_metrics(name):
    with open(f"results/metrics/{name}.json", "r") as f:
        return json.load(f)

def save_predictions(name, y_pred):
    np.save(f"results/predictions/{name}_preds.npy", y_pred)

def load_predictions(name):
    return np.load(f"results/predictions/{name}_preds.npy")