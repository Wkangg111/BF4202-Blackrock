"""Probability calibration for a multi-class LinearSVC.

The SVM's one-vs-rest margins are turned into probabilities with a temperature-scaled softmax:
    p(class | x) = softmax(margin(x) / T)
T is a single number fitted (by log-loss) on out-of-fold margins from held-out-template cross-validation,
so the probabilities reflect how sure the model should be on wording it has not seen before.
Kept in its own module so the saved joblib model can be loaded by predict.py.
"""
import numpy as np
from scipy.optimize import minimize_scalar


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit_temperature(margins, y_idx):
    """Return T minimising the mean negative log-likelihood of softmax(margins / T)."""
    rows = np.arange(len(y_idx))
    def nll(log_t):
        p = softmax(margins / np.exp(log_t))
        return -np.mean(np.log(np.clip(p[rows, y_idx], 1e-12, 1)))
    return float(np.exp(minimize_scalar(nll, bounds=(-6, 4), method='bounded').x))


class TemperatureScaledSVM:
    """A fitted LinearSVC plus a temperature; exposes classes_, decision_function, predict, predict_proba."""
    def __init__(self, svm, temperature):
        self.svm, self.temperature = svm, temperature
        self.classes_ = svm.classes_

    def decision_function(self, X):
        return self.svm.decision_function(X)

    def predict(self, X):
        return self.svm.predict(X)

    def predict_proba(self, X):
        return softmax(self.decision_function(X) / self.temperature)
