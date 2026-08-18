"""Configuration module for OpenClaw Credit Risk Agent."""

import os
from pathlib import Path
from functools import lru_cache
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


def _resolve_scorecard_path() -> Path:
    """Resolve scorecard path with hierarchical fallback."""
    env_path = os.getenv("SCORECARD_PATH")
    if env_path and Path(env_path).exists():
        return Path(env_path)

    agent_root = Path(__file__).resolve().parent.parent
    # 1. Local bundled repo data directory
    local_path = agent_root / "data" / "df_scorecard.csv"
    if local_path.exists():
        return local_path

    # 2. Local repo root fallback
    local_root_path = agent_root / "df_scorecard.csv"
    if local_root_path.exists():
        return local_root_path

    # 3. Parent workspace fallback
    parent_path = agent_root.parent / "df_scorecard.csv"
    if parent_path.exists():
        return parent_path

    return local_path


def _resolve_ml_output_dir() -> Path:
    """Resolve ML challenger output directory with hierarchical fallback."""
    env_dir = os.getenv("ML_OUTPUT_DIR")
    if env_dir and Path(env_dir).exists():
        return Path(env_dir)

    agent_root = Path(__file__).resolve().parent.parent
    # 1. Local bundled repo data directory
    local_ml_dir = agent_root / "data" / "ml_output"
    if local_ml_dir.exists():
        return local_ml_dir

    # 2. Local ml_output_dir at agent root
    local_alt_dir = agent_root / "ml_challenger_output"
    if local_alt_dir.exists():
        return local_alt_dir

    # 3. Parent workspace fallback
    parent_ml_dir = agent_root.parent / "ml_challenger_output"
    if parent_ml_dir.exists():
        return parent_ml_dir

    return local_ml_dir


class Settings(BaseModel):
    """Application and Agent Settings."""

    # Project directories
    agent_root: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    parent_project_root: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent)

    # Model artifact paths
    scorecard_path: Path = Field(default_factory=_resolve_scorecard_path)
    ml_output_dir: Path = Field(default_factory=_resolve_ml_output_dir)
    audit_log_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "audit_logs")

    # Groq API Configuration
    groq_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    groq_model: str = Field(default_factory=lambda: os.getenv("GROQ_MODEL", "deepseek-r1-distill-llama-70b"))
    groq_temperature: float = Field(default_factory=lambda: float(os.getenv("GROQ_TEMPERATURE", "0.1")))
    groq_max_tokens: int = Field(default_factory=lambda: int(os.getenv("GROQ_MAX_TOKENS", "2048")))

    # Environment
    openclaw_env: str = Field(default_factory=lambda: os.getenv("OPENCLAW_ENV", "development"))
    log_level: str = Field(default_factory=lambda: os.getenv("OPENCLAW_LOG_LEVEL", "INFO"))

    # Server settings
    host: str = Field(default_factory=lambda: os.getenv("HOST", "0.0.0.0"))
    port: int = Field(default_factory=lambda: int(os.getenv("PORT", "8000")))

    # Underwriting Policy Rules
    policy_auto_approve_max_pd: float = 0.05
    policy_auto_reject_min_pd: float = 0.25
    policy_min_credit_score: int = 580
    policy_max_dti: float = 45.0
    policy_max_loan_amount_auto: float = 40000.0

    # Macroeconomic Stress Testing Scenarios
    stress_scenarios: dict = Field(default_factory=lambda: {
        "baseline": {"gdp_shock_pct": 0.0, "unemp_shock_pct": 0.0, "pd_multiplier": 1.0, "lgd_multiplier": 1.0},
        "mild_downturn": {"gdp_shock_pct": -1.0, "unemp_shock_pct": 2.0, "pd_multiplier": 1.35, "lgd_multiplier": 1.15},
        "severe_recession": {"gdp_shock_pct": -2.0, "unemp_shock_pct": 5.0, "pd_multiplier": 1.85, "lgd_multiplier": 1.30},
    })

    model_config = ConfigDict(arbitrary_types_allowed=True)


@lru_cache()
def get_settings() -> Settings:
    """Retrieve cached singleton settings."""
    # Ensure audit directory exists
    settings = Settings()
    settings.audit_log_dir.mkdir(parents=True, exist_ok=True)
    return settings
