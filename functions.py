import numpy as np


def exp(x):
    y = np.exp(x)
    y_scaled = y / max(y)
    return y_scaled


def polinom(x, coefs:list):
    y = np.polyval(coefs, x)
    y_scaled = y / max(y)
    y_clipped = np.clip(y_scaled, 0, 1)
    return y_clipped


def gauss(x, coefs:list):
    y = 1 / (2 * np.pi * coefs[0]) * np.exp(-(x - coefs[1])**2 / (2 * coefs[0]**2))
    y_scaled = y / max(y)
    return y_scaled
