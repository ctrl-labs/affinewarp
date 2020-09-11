"""
Tests that optimization is being performed correctly.
"""

import numpy as np
from numpy.testing import assert_allclose

from affinewarp import PiecewiseWarping


def test_identity_warps_gauss_errors():
    """
    Test that model fits converge to trial-average with identity warps.
    """

    # Create small, synthetic dataset
    n_trials = 10
    n_timepoints = 11
    n_units = 3
    data = np.random.randn(n_trials, n_timepoints, n_units)

    # Define model
    model = PiecewiseWarping(
        n_knots=0,
        warp_reg_scale=0.0,
        smoothness_reg_scale=0.0,
        l2_reg_scale=0.0,
    )

    # Fit model template without updating warps.
    model.initialize_warps(n_trials)
    model._fit_template(data, np.arange(n_trials))

    assert_allclose(model.template, data.mean(axis=0), rtol=1e-4)
