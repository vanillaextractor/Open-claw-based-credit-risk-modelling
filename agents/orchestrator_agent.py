"""Orchestrator Agent for End-to-End Multi-Agent Credit Risk Assessment."""

import logging
from typing import Dict, Any, Union
from datetime import datetime, timezone
import uuid

from openclaw_credit_risk_agent.agents.base_agent import BaseAgent
from openclaw_credit_risk_agent.agents.data_agent import DataAgent
from openclaw_credit_risk_agent.agents.risk_model_agent import RiskModelAgent
from openclaw_credit_risk_agent.agents.model_comparison_agent import ModelComparisonAgent
from openclaw_credit_risk_agent.agents.explainability_agent import ExplainabilityAgent
from openclaw_credit_risk_agent.agents.stress_testing_agent import StressTestingAgent
from openclaw_credit_risk_agent.agents.monitoring_agent import MonitoringAgent
from openclaw_credit_risk_agent.agents.policy_agent import PolicyAgent

from openclaw_credit_risk_agent.tools.schemas import ApplicantData, FullCreditAssessment
from openclaw_credit_risk_agent.tools.audit_logger import get_audit_logger

logger = logging.getLogger("openclaw_credit_risk.orchestrator")

ROLE_PROMPT = """You are the Senior Chief Risk Officer and Orchestrator Agent in an OpenClaw Credit Risk Assessment framework.
Your task is to synthesize the findings from all specialized credit risk agents into a coherent,
rigorous, and actionable Credit Risk Assessment Memorandum.
Include:
1. Executive Decision Summary
2. Quantitative Expected Loss Breakdown (PD, LGD, EAD, EL)
3. Model Consensus & Benchmark Analysis (Scorecard vs ML Challenger)
4. Primary Risk Drivers & Adverse Action Factors
5. Resilience under Macroeconomic Stress
6. Final Underwriting Policy Recommendation & Required Stated Conditions

MANDATORY: You must not alter any numerical values produced by the deterministic engine."""


class OrchestratorAgent(BaseAgent):
    """Main OpenClaw orchestrator managing multi-agent collaboration."""

    def __init__(self):
        super().__init__(name="OrchestratorAgent", role_prompt=ROLE_PROMPT)
        self.data_agent = DataAgent()
        self.risk_model_agent = RiskModelAgent()
        self.comparison_agent = ModelComparisonAgent()
        self.explainability_agent = ExplainabilityAgent()
        self.stress_agent = StressTestingAgent()
        self.monitoring_agent = MonitoringAgent()
        self.policy_agent = PolicyAgent()
        self.audit_logger = get_audit_logger()

    def assess_applicant(self, raw_input: Union[Dict[str, Any], ApplicantData]) -> FullCreditAssessment:
        """Run complete multi-agent assessment workflow with audit logging."""
        tool_calls = []

        # 1. Data Agent: Validate & Cleanse
        if isinstance(raw_input, ApplicantData):
            applicant = raw_input
            data_summary = {"is_valid": True, "anomalies_detected": []}
        else:
            applicant, data_summary = self.data_agent.validate_and_prepare(raw_input)
        tool_calls.append({"agent": "DataAgent", "action": "validate_and_prepare", "status": "SUCCESS"})

        # 2. Risk Model Agent: PD, LGD, EAD, EL
        risk_output = self.risk_model_agent.run_assessment(applicant)
        tool_calls.append({"agent": "RiskModelAgent", "action": "run_assessment", "status": "SUCCESS"})

        # 3. Model Comparison Agent: Challenger Models
        comparison_res = self.comparison_agent.compare(applicant)
        challenger_pd = next((m.predicted_pd for m in comparison_res.models if "XGBoost" in m.model_name), risk_output["pd"])
        tool_calls.append({"agent": "ModelComparisonAgent", "action": "compare", "status": "SUCCESS"})

        # 4. Explainability Agent: SHAP Drivers
        shap_res = self.explainability_agent.explain(applicant, top_n=5)
        tool_calls.append({"agent": "ExplainabilityAgent", "action": "explain", "status": "SUCCESS"})

        # 5. Stress Testing Agent: Macroeconomic Scenarios
        stress_res = self.stress_agent.stress_test(applicant, scenario="all")
        tool_calls.append({"agent": "StressTestingAgent", "action": "stress_test", "status": "SUCCESS"})

        # 6. Policy Agent: Underwriting Rules
        policy_res = self.policy_agent.evaluate(
            applicant,
            pd_val=risk_output["pd"],
            score=risk_output["credit_score"],
            el_val=risk_output["expected_loss"]
        )
        tool_calls.append({"agent": "PolicyAgent", "action": "evaluate", "status": "SUCCESS"})

        # 7. LLM Narrative Synthesis (using Groq if available)
        assessment_id = f"CR-{uuid.uuid4().hex[:8].upper()}"
        prompt = (
            f"Generate an executive credit memo for Application ID {applicant.application_id or assessment_id}.\n"
            f"Applicant Data: Loan Amount=${applicant.loan_amnt:,.0f}, Income=${applicant.annual_inc:,.0f}, DTI={applicant.dti}%, Grade={applicant.grade}, Home={applicant.home_ownership}, Purpose={applicant.purpose}.\n"
            f"Model Outputs: Score={risk_output['credit_score']}, Grade={risk_output['risk_grade']}, PD={risk_output['pd']:.2%}, LGD={risk_output['lgd']:.2%}, EAD=${risk_output['ead']:,.2f}, EL=${risk_output['expected_loss']:,.2f} ({risk_output['el_rate_pct']:.2f}%).\n"
            f"Challenger Model: XGBoost PD={challenger_pd:.2%}.\n"
            f"Policy Decision: {policy_res.decision.value}. Reasons: {', '.join(policy_res.reasons)}.\n"
            f"Top Risk Drivers: {', '.join([f'{d.feature_label} ({d.direction})' for d in shap_res.top_risk_drivers])}.\n"
            f"Worst-case Stressed EL: ${stress_res.worst_case_el:,.2f}.\n"
            f"Provide a concise, 3-4 paragraph professional risk assessment memorandum."
        )
        narrative_summary = self.query_llm(prompt)

        # 8. Assemble Structured Result
        stress_summary_dict = {s.scenario_name: s.stressed_el for s in stress_res.scenarios}

        assessment = FullCreditAssessment(
            assessment_id=assessment_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            application_id=applicant.application_id,
            pd=risk_output["pd"],
            lgd=risk_output["lgd"],
            ead=risk_output["ead"],
            expected_loss=risk_output["expected_loss"],
            credit_score=risk_output["credit_score"],
            risk_grade=risk_output["risk_grade"],
            model="logistic_scorecard",
            challenger_pd=challenger_pd,
            top_risk_drivers=shap_res.top_risk_drivers,
            policy_decision=policy_res.decision,
            policy_reasons=policy_res.reasons,
            model_comparison=comparison_res,
            stress_test_summary=stress_summary_dict,
            explanation_summary=narrative_summary or (
                f"Applicant evaluated under Basel Scorecard: Score {risk_output['credit_score']} (Grade {risk_output['risk_grade']}), "
                f"PD {risk_output['pd']:.2%}, LGD {risk_output['lgd']:.1%}, EAD ${risk_output['ead']:,.2f}, Expected Loss ${risk_output['expected_loss']:,.2f}. "
                f"Policy Decision: {policy_res.decision.value}."
            )
        )

        # 9. Log Immutable Audit Record
        try:
            self.audit_logger.log_assessment(
                assessment_id=assessment_id,
                application_data=applicant.model_dump(),
                model_outputs=assessment.model_dump(),
                policy_result=policy_res.model_dump(),
                tool_calls=tool_calls,
                agent_narrative=narrative_summary
            )
        except Exception as e:
            logger.warning(f"Audit log writing failed: {e}")

        return assessment


# Global singleton
_orchestrator = None

def get_orchestrator() -> OrchestratorAgent:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = OrchestratorAgent()
    return _orchestrator
