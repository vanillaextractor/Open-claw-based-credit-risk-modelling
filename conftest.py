"""Pytest configuration and environment fixtures."""

import sys
import types
from pathlib import Path

# Set up paths so tests work standalone from any directory
ROOT_DIR = Path(__file__).resolve().parent
PARENT_DIR = ROOT_DIR.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

# Ensure 'openclaw_credit_risk_agent' package namespace is always resolvable
if "openclaw_credit_risk_agent" not in sys.modules:
    pkg = types.ModuleType("openclaw_credit_risk_agent")
    pkg.__path__ = [str(ROOT_DIR)]
    pkg.__file__ = str(ROOT_DIR / "__init__.py")
    sys.modules["openclaw_credit_risk_agent"] = pkg
