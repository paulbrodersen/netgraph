#!/usr/bin/env python
"""
Test _utils.py
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt

from utils import (
    _resample_spline,
)

np.random.seed(42)

def test_resample_spline():
    spline = np.array([
        (0, 0),
        (1, 1)
    ])
    total_points = 100
    resampled = _resample_spline(spline, total_points)
    x = np.linspace(*spline[:, 0], total_points)
    y = np.linspace(*spline[:, 1], total_points)
    theoretical = np.c[x, y]
    assert np.all(np.isclose(resampled, theoretical))
