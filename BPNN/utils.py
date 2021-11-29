import numpy as np

_abs = abs


def bipolar(vector):
    """
    Bipolar represents the mapping from binary values (alphabet {0, 1}) to bipolar values (alphabet {1, -1}):
    - 0 -> 1
    - 1 -> -1
    """
    return np.where(vector == 0, np.ones(vector.shape), -np.ones(vector.shape))


def bin(vector):
    """
    Bin represents the mapping from bipolar values (alphabet {-1, -1}) to binary values (alphabet {0, 1}):
    - -1 -> 1
    - 1 -> 0
    """
    return np.where(vector == 1, np.zeros(vector.shape), np.ones(vector.shape))


def sign(vector):
    """
    Sign represents the mapping from integer to their sign representation:
    - x | x >= 0 -> 1
    - else -> -1
    """
    return np.where(vector >= 0, np.ones(vector.shape), -np.ones(vector.shape))


def abs(vector):
    """
    Abs represents the mapping from integers to their positive counterpart in case they are negative:
    - x | x <= 0 -> -x
    - else -> x
    """
    return np.array(list(map(_abs, vector)))
