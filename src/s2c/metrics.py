#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
metrics.py

Description: This module implements the Euclidean and Manhattan distance metrics.

Author: Aldo Tapia
Date: 2024-10-18
"""

import numpy as np

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
