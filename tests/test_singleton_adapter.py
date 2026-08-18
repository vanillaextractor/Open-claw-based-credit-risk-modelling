"""Tests for CreditModelAdapter Singleton Pattern and Failure Recovery."""

import pytest
from pathlib import Path
from unittest.mock import patch

from openclaw_credit_risk_agent.tools.credit_model_adapter import (
    CreditModelAdapter,
    get_credit_adapter,
)


def test_singleton_same_instance():
    """Verify that multiple calls to CreditModelAdapter() return the identical instance."""
    CreditModelAdapter.reset_instance()
    adapter1 = CreditModelAdapter()
    adapter2 = CreditModelAdapter()
    assert adapter1 is adapter2
    assert hasattr(adapter1, "scorecard_df")
    assert adapter1.scorecard_df is not None


def test_get_credit_adapter_helper_and_reload():
    """Verify get_credit_adapter() behaves as singleton and supports reload."""
    CreditModelAdapter.reset_instance()
    adapter1 = get_credit_adapter()
    adapter2 = get_credit_adapter()
    assert adapter1 is adapter2

    adapter3 = get_credit_adapter(reload=True)
    assert adapter3 is not None
    assert hasattr(adapter3, "scorecard_df")


def test_initialization_failure_does_not_cache_broken_instance():
    """Verify that if initialization fails (e.g. FileNotFoundError),
    no broken instance is cached, and subsequent calls retry cleanly rather than
    raising confusing 'AttributeError: object has no attribute scorecard_df'.
    """
    CreditModelAdapter.reset_instance()

    # Simulate missing scorecard file causing FileNotFoundError during _initialize()
    with patch.object(CreditModelAdapter, "_load_scorecard", side_effect=FileNotFoundError("Scorecard not found at /path/df_scorecard.csv")):
        # First call must raise FileNotFoundError
        with pytest.raises(FileNotFoundError) as exc_info1:
            CreditModelAdapter()
        assert "Scorecard not found" in str(exc_info1.value)

        # Second call must ALSO raise FileNotFoundError (NOT AttributeError on half-built instance)
        with pytest.raises(FileNotFoundError) as exc_info2:
            CreditModelAdapter()
        assert "Scorecard not found" in str(exc_info2.value)

    # After error is resolved, initialization succeeds cleanly
    CreditModelAdapter.reset_instance()
    adapter_recovered = CreditModelAdapter()
    assert adapter_recovered is not None
    assert hasattr(adapter_recovered, "scorecard_df")
    assert not adapter_recovered.scorecard_df.empty
