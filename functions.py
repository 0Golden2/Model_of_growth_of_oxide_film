import numpy as np


def exp(x, norm=True):
    y = np.exp(x)
    if norm:
        y = y / max(y)
    return y


def polinom(x, coefs:list, norm=True):
    y = np.polyval(coefs, x)
    if norm:
        y_scaled = y / max(y)
        y = np.clip(y_scaled, 0, 1)
    return y


def gauss(x, coefs:list, norm=True):
    y = 1 / (2 * np.pi * coefs[0]) * np.exp(-(x - coefs[1])**2 / (2 * coefs[0]**2))
    if norm:
        y = y / max(y)
    return y
