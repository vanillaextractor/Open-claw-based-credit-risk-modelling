"""Audit Logger for Credit Risk Assessment Layer.

Provides immutable, structured audit trails for all model and agent operations.
Masks API keys and secrets before saving.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

from openclaw_credit_risk_agent.config.settings import get_settings

logger = logging.getLogger("openclaw_credit_risk.audit")


class AuditLogger:
    """Handles structured audit logging for credit decisions."""

    def __init__(self):
        self.settings = get_settings()
        self.log_dir = self.settings.audit_log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def log_assessment(
        self,
        assessment_id: str,
        application_data: Dict[str, Any],
        model_outputs: Dict[str, Any],
        policy_result: Dict[str, Any],
        tool_calls: Optional[list] = None,
        agent_narrative: Optional[str] = None
    ) -> Path:
        """Write an audit record to disk in JSON format."""
        # Sanitize application data (strip any potential secrets)
        sanitized_input = self._sanitize_dict(application_data)
        sanitized_outputs = self._sanitize_dict(model_outputs)
        sanitized_policy = self._sanitize_dict(policy_result)

        now_utc = datetime.now(timezone.utc)

        record = {
            "audit_version": "1.0.0",
            "timestamp": now_utc.isoformat(),
            "assessment_id": assessment_id,
            "application_id": sanitized_input.get("application_id"),
            "model_metadata": {
                "pd_model": "WoE Logistic Scorecard v1.0",
                "lgd_model": "Two-Stage Hurdle Recovery Model v1.0",
                "ead_model": "Credit Conversion Factor Linear Model v1.0",
                "challenger_model": "XGBoost Classifier (300 estimators, max_depth=6)",
                "policy_engine_version": sanitized_policy.get("policy_version", "v1.0")
            },
            "applicant_features_snapshot": sanitized_input,
            "deterministic_results": {
                "pd": sanitized_outputs.get("pd"),
                "lgd": sanitized_outputs.get("lgd"),
                "ead": sanitized_outputs.get("ead"),
                "expected_loss": sanitized_outputs.get("expected_loss"),
                "credit_score": sanitized_outputs.get("credit_score"),
                "risk_grade": sanitized_outputs.get("risk_grade"),
                "challenger_pd": sanitized_outputs.get("challenger_pd"),
            },
            "policy_decision": {
                "decision": sanitized_policy.get("decision"),
                "reasons": sanitized_policy.get("reasons", []),
                "conditions": sanitized_policy.get("recommended_conditions", [])
            },
            "shap_risk_drivers": sanitized_outputs.get("top_risk_drivers", []),
            "stress_test_summary": sanitized_outputs.get("stress_test_summary", {}),
            "tool_executions": tool_calls or [],
            "agent_narrative_summary": agent_narrative or "",
        }

        filename = f"audit_{assessment_id}_{now_utc.strftime('%Y%m%d_%H%M%S')}.json"
        target_path = self.log_dir / filename

        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, default=str)

        logger.info(f"Audit log created: {target_path}")
        return target_path

    def _sanitize_dict(self, data: Any) -> Any:
        """Recursively remove API keys or sensitive credential patterns."""
        if isinstance(data, dict):
            clean = {}
            for k, v in data.items():
                if any(sec in k.lower() for sec in ["api_key", "secret", "password", "token", "auth"]):
                    clean[k] = "[MASKED_SECRET]"
                else:
                    clean[k] = self._sanitize_dict(v)
            return clean
        elif isinstance(data, list):
            return [self._sanitize_dict(item) for item in data]
        return data


# Global singleton
_audit_logger = None

def get_audit_logger() -> AuditLogger:
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger
