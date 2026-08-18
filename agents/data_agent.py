"""Data Agent for Applicant Schema Validation and Pre-processing."""

from typing import Dict, Any, Tuple
from openclaw_credit_risk_agent.agents.base_agent import BaseAgent
from openclaw_credit_risk_agent.tools.schemas import ApplicantData

ROLE_PROMPT = """You are the Data Agent in an OpenClaw Credit Risk Assessment framework.
Your role is to validate incoming borrower information, detect data anomalies or missing fields,
and confirm that all attributes conform to Basel III / Lending Club schema constraints."""


class DataAgent(BaseAgent):
    """Validates and prepares applicant payloads for model consumption."""

    def __init__(self):
        super().__init__(name="DataAgent", role_prompt=ROLE_PROMPT)

    def validate_and_prepare(self, raw_input: Dict[str, Any]) -> Tuple[ApplicantData, Dict[str, Any]]:
        """Validate input dictionary against Pydantic schema and check for business anomalies."""
        import copy
        data = copy.deepcopy(raw_input)  # Never mutate caller's dict
        anomalies = []

        # 1. Normalize loan amount
        loan_amnt = data.get("loan_amnt")
        if loan_amnt is None or float(loan_amnt) <= 0:
            anomalies.append("Missing or invalid loan amount; defaulting to $10,000")
            data["loan_amnt"] = 10000.0

        # 2. Check annual income
        annual_inc = data.get("annual_inc")
        if annual_inc is None or float(annual_inc) <= 0:
            anomalies.append("Annual income missing or zero; defaulting to median $50,000")
            data["annual_inc"] = 50000.0

        # 3. Check DTI
        dti = data.get("dti")
        if dti is None:
            data["dti"] = 18.0
            anomalies.append("DTI missing; defaulted to 18.0%")

        # 4. Check Interest Rate
        int_rate = data.get("int_rate")
        if int_rate is None or float(int_rate) <= 0:
            data["int_rate"] = 12.0
            anomalies.append("Interest rate missing; defaulted to benchmark 12.0%")

        # Instantiate validated schema
        applicant = ApplicantData(**data)

        summary = {
            "is_valid": True,
            "anomalies_detected": anomalies,
            "applicant_id": applicant.application_id,
            "features_count": len(data)
        }
        return applicant, summary
