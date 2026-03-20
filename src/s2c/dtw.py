#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dtw.py

Description: This module implements the Dynamic Time Warping (DTW) algorithm.

Author: Aldo Tapia
Date: 2024-11-21
"""

import numpy as np
import math


def euclidean(x: np.ndarray, y: np.ndarray) -> float:
    """
    This function computes the Euclidean distance between two values.

    Parameters
    ----------
    x : float
        A value.
    y : float
        A value.

    Returns
    -------
    float
        The Euclidean distance between the two values.
    """
    res = None
    if x.size != y.size:
        raise ValueError('The two sequences must have the same length.')
    if x.size == 0:
        raise ValueError('The sequences must not be empty.')
    if x.size == 1:
        res = np.sqrt((x - y) ** 2)
    if x.size > 1:
        res = np.sqrt(np.sum((x - y) ** 2))
    return res

def manhattan(x: np.ndarray, y: np.ndarray) -> float:
    """
    This function computes the Manhattan distance between two values.

    Parameters
    ----------
    x : float
        A value.
    y : float
        A value.

    Returns
    -------
    float
        The Manhattan distance between the two values.
    """
    res = None
    if x.size != y.size:
        raise ValueError('The two sequences must have the same length.')
    if x.size == 0:
        raise ValueError('The sequences must not be empty.')
    if x.size == 1:
        res = np.abs(x - y)
    if x.size > 1:
        res = np.sum(np.abs(x - y))
    return res

def g(
    ti: int,
    tj: int
    ) -> int:
    """
    This function computes the temporal gap between two dates.
    As proposed by Maus et al. (2015).

    Equation:

    .. math::
        g(t_i,t_j) = |t_i - t_j|


    Parameters
    ----------
    t1 : int
        integer date representation for serie i.
    t2 : datetime
        integer date representation for serie j.

    Returns
    -------
    int
        The temporal gap between the two dates.
    """
    return abs(ti - tj)

def logi_omega(
    ti: int,
    tj: int,
    alpha: float,
    beta: float
    ) -> float:
    """
    This function computes the logistic temporal cost between two dates (omega).
    As proposed by Maus et al. (2015).

    Equation:

    .. math::
        \omega_{i,j} = \frac{1}{1 + e^{-\alpha(g(t_i,t_j) - \beta)}}


    Parameters
    ----------
    t1 : int
        integer date representation for serie i.
    t2 : datetime
        integer date representation for serie j.
    alpha : float
        alpha steepness parameter.
    beta : float
        beta midpoint parameter.

    Returns
    -------
    float
        The logistic temporal cost between the two dates.
    """
    return 1 / (1 + math.exp(-alpha * (g(ti,tj) - beta)))

def matrix_distance(
    x: np.ndarray,
    y: np.ndarray,
    dissimilarity: int,
    tx: np.ndarray = None,
    ty: np.ndarray  = None,
    alpha: float = 0.1,
    beta: float = 100
    ) -> np.ndarray:
    """
    This function computes the distance matrix between two time series.
    
    Parameters
    ----------
    x : np.ndarray
        Time series x.
    y : np.ndarray
        Time series y.
    dissimilarity : int
        Dissimilarity metric to be used. 0 for Euclidean, 1 for Manhattan.
    tx : np.ndarray
        Time series x timestamps.
    ty : np.ndarray
        Time series y timestamps.
    alpha : float
        alpha steepness parameter.
    beta : float
        beta midpoint parameter.

    Returns
    -------
    matrix : np.ndarray
        Distance matrix.
    """
    n = len(x)
    m = len(y)
    matrix = np.zeros((n, m), dtype=np.float64)

    if (tx is not None) and (ty is not None):
        for i in range(n):
            for j in range(m):
                if dissimilarity == 0:  # Euclidean
                    matrix[i, j] = euclidean(x[i], y[j]) + logi_omega(tx[i], ty[j], alpha, beta)
                elif dissimilarity == 1:  # Manhattan
                    matrix[i, j] = manhattan(x[i], y[j]) + logi_omega(tx[i], ty[j], alpha, beta)
                else:
                    raise ValueError("Invalid distance metric. Use 0 for 'euclidean' or 1 for 'manhattan'.")
    else:
        for i in range(n):
            for j in range(m):
                if dissimilarity == 0:  # Euclidean
                    matrix[i, j] = euclidean(x[i], y[j])
                elif dissimilarity == 1:  # Manhattan
                    matrix[i, j] = manhattan(x[i], y[j])
                else:
                    raise ValueError("Invalid distance metric. Use 0 for 'euclidean' or 1 for 'manhattan'.")

    return matrix

def dtw(
    x: np.ndarray,
    y: np.ndarray,
    dissimilarity: int,
    tx: np.ndarray = None,
    ty: np.ndarray  = None,
    alpha: float = 0.1,
    beta: float = 100
    ) -> np.ndarray:
    """
    This function computes either Dynamic Time Warping (DTW) or
    Time Weighted Dynamic Time Warping (TWDTW) distance between two
    sequences.
    
    TWDTW as proposed by Maus et al. (2015).
    
    Parameters
    ----------
    x : np.ndarray
        Time series x.
    y : np.ndarray
        Time series y.
    dissimilarity : int
        Dissimilarity metric to be used. 0 for Euclidean, 1 for Manhattan.
    tx : np.ndarray
        Time series x timestamps.
    ty : np.ndarray
        Time series y timestamps.
    alpha : float
        alpha steepness parameter.
    beta : float
        beta midpoint parameter.

    Returns
    -------
    float
        The DTW/TWDTW distance between the two sequences.
    """
    matrix = matrix_distance(x, y, dissimilarity, tx, ty, alpha, beta)

    n = matrix.shape[0]
    m = matrix.shape[1]

    acc_cost = np.zeros((n, m), dtype=np.float64)

    acc_cost[0, 0] = matrix[0, 0]
    for i in range(1, n):
        acc_cost[i, 0] = acc_cost[i - 1, 0] + matrix[i, 0]
    for j in range(1, m):
        acc_cost[0, j] = acc_cost[0, j - 1] + matrix[0, j]
    for i in range(1, n):
        for j in range(1, m):
            acc_cost[i, j] = matrix[i, j] + min(acc_cost[i - 1, j],
                                                acc_cost[i, j - 1],
                                                acc_cost[i - 1, j - 1])

    return acc_cost[n - 1, m - 1]
