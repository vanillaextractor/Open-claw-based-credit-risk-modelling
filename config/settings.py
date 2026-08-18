"""Configuration module for OpenClaw Credit Risk Agent."""

import os
from pathlib import Path
from functools import lru_cache
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


class Settings(BaseModel):
    """Application and Agent Settings."""

    # Project directories
    agent_root: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    parent_project_root: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent)

    # Model artifact paths
    scorecard_path: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent / "df_scorecard.csv")
    ml_output_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent / "ml_challenger_output")
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
