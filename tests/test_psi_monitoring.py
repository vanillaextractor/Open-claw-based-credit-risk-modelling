"""Tests for Population Stability Index (PSI) Engine."""

import numpy as np
import pytest
from openclaw_credit_risk_agent.tools.credit_model_adapter import get_credit_adapter


@pytest.fixture
def adapter():
    return get_credit_adapter()


def test_psi_identical_distributions(adapter):
    """Verify that identical distributions yield PSI close to zero (< 0.01)."""
    np.random.seed(42)
    expected = np.random.normal(loc=100, scale=15, size=1000)
    actual = expected.copy()

    psi_val = adapter.calculate_psi(expected, actual, n_bins=10)
    assert psi_val < 0.01


def test_psi_significant_drift(adapter):
    """Verify that shifted distributions yield high PSI (> 0.25)."""
    np.random.seed(42)
    expected = np.random.normal(loc=100, scale=15, size=1000)
    actual = np.random.normal(loc=140, scale=25, size=1000)  # substantial shift

    psi_val = adapter.calculate_psi(expected, actual, n_bins=10)
    assert psi_val > 0.25


def test_precomputed_psi_summary(adapter):
    """Verify loading of precomputed feature PSI summaries."""
    psi_res = adapter.get_psi_summary()
    assert psi_res.overall_status in ["STABLE", "INVESTIGATE"]
    assert psi_res.stable_features > 0
