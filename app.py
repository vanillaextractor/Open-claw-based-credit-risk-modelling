#!/usr/bin/env python3
"""OpenClaw Credit Risk Agent CLI Application.

Interactive command-line interface for credit risk assessment, multi-model comparison,
SHAP explainability, macroeconomic stress testing, and underwriting policy decisions.
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any

import types

# Ensure project root is in sys.path and package namespace resolves
AGENT_ROOT = Path(__file__).resolve().parent
PARENT_ROOT = AGENT_ROOT.parent
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))
if str(PARENT_ROOT) not in sys.path:
    sys.path.insert(0, str(PARENT_ROOT))

if "openclaw_credit_risk_agent" not in sys.modules:
    _pkg = types.ModuleType("openclaw_credit_risk_agent")
    _pkg.__path__ = [str(AGENT_ROOT)]
    _pkg.__file__ = str(AGENT_ROOT / "__init__.py")
    sys.modules["openclaw_credit_risk_agent"] = _pkg

from openclaw_credit_risk_agent.agents.orchestrator_agent import get_orchestrator
from openclaw_credit_risk_agent.tools.schemas import ApplicantData, FullCreditAssessment


# Pre-defined sample profiles for testing
SAMPLE_PROFILES: Dict[str, Dict[str, Any]] = {
    "low_risk": {
        "application_id": "APP-LOW-001",
        "loan_amnt": 15000.0,
        "funded_amnt": 15000.0,
        "int_rate": 7.25,
        "grade": "A",
        "sub_grade": "A2",
        "emp_length": "8 years",
        "home_ownership": "MORTGAGE",
        "annual_inc": 115000.0,
        "verification_status": "Source Verified",
        "purpose": "debt_consolidation",
        "addr_state": "CA",
        "dti": 11.4,
        "delinq_2yrs": 0.0,
        "inq_last_6mths": 0.0,
        "mths_since_last_delinq": None,
        "mths_since_last_record": None,
        "open_acc": 14.0,
        "pub_rec": 0.0,
        "revol_bal": 12000.0,
        "revol_util": 24.5,
        "total_acc": 28.0,
        "initial_list_status": "w",
        "term": "36 months",
        "collections_12_mths_ex_med": 0.0,
        "acc_now_delinq": 0.0,
        "tot_coll_amt": 0.0,
        "tot_cur_bal": 185000.0,
        "total_rev_hi_lim": 50000.0,
        "mths_since_earliest_cr_line": 210.0,
        "mths_since_issue_d": 24.0,
    },
    "borderline": {
        "application_id": "APP-MID-002",
        "loan_amnt": 22000.0,
        "funded_amnt": 22000.0,
        "int_rate": 13.85,
        "grade": "C",
        "sub_grade": "C3",
        "emp_length": "3 years",
        "home_ownership": "RENT",
        "annual_inc": 62000.0,
        "verification_status": "Not Verified",
        "purpose": "credit_card",
        "addr_state": "TX",
        "dti": 21.8,
        "delinq_2yrs": 1.0,
        "inq_last_6mths": 2.0,
        "mths_since_last_delinq": 18.0,
        "mths_since_last_record": None,
        "open_acc": 9.0,
        "pub_rec": 0.0,
        "revol_bal": 18500.0,
        "revol_util": 58.0,
        "total_acc": 16.0,
        "initial_list_status": "f",
        "term": "36 months",
        "collections_12_mths_ex_med": 0.0,
        "acc_now_delinq": 0.0,
        "tot_coll_amt": 0.0,
        "tot_cur_bal": 42000.0,
        "total_rev_hi_lim": 32000.0,
        "mths_since_earliest_cr_line": 155.0,
        "mths_since_issue_d": 36.0,
    },
    "high_risk": {
        "application_id": "APP-HIGH-003",
        "loan_amnt": 35000.0,
        "funded_amnt": 35000.0,
        "int_rate": 22.45,
        "grade": "F",
        "sub_grade": "F4",
        "emp_length": "< 1 year",
        "home_ownership": "RENT",
        "annual_inc": 38000.0,
        "verification_status": "Verified",
        "purpose": "small_business",
        "addr_state": "FL",
        "dti": 37.5,
        "delinq_2yrs": 3.0,
        "inq_last_6mths": 5.0,
        "mths_since_last_delinq": 6.0,
        "mths_since_last_record": 24.0,
        "open_acc": 6.0,
        "pub_rec": 2.0,
        "revol_bal": 31000.0,
        "revol_util": 89.2,
        "total_acc": 12.0,
        "initial_list_status": "f",
        "term": "60 months",
        "collections_12_mths_ex_med": 1.0,
        "acc_now_delinq": 1.0,
        "tot_coll_amt": 1200.0,
        "tot_cur_bal": 31000.0,
        "total_rev_hi_lim": 35000.0,
        "mths_since_earliest_cr_line": 95.0,
        "mths_since_issue_d": 48.0,
    }
}


def print_banner():
    print("\n" + "═" * 78)
    print("  🚀 OPENCLAW CREDIT RISK AGENT — ENTERPRISE ASSESSMENT SYSTEM")
    print("  Basel III / IFRS 9 Compliant Expected Loss & ML Challenger Engine")
    print("═" * 78)


def display_assessment(res: FullCreditAssessment):
    """Format and print structured assessment memo."""
    print("\n" + "┌" + "─" * 76 + "┐")
    print(f"│  CREDIT RISK ASSESSMENT MEMORANDUM — ID: {res.assessment_id:<32} │")
    print(f"│  Timestamp: {res.timestamp:<62} │")
    print("├" + "─" * 76 + "┤")

    # Core Metrics
    print("│  QUANTITATIVE MODEL RESULTS (Source of Truth):                             │")
    print(f"│    • Probability of Default (PD) : {res.pd:>8.2%}  (Grade {res.risk_grade}, Score: {res.credit_score})              │")
    print(f"│    • Loss Given Default (LGD)    : {res.lgd:>8.2%}                                         │")
    print(f"│    • Exposure at Default (EAD)   : ${res.ead:>10,.2f}                                      │")
    print(f"│    • Expected Loss (EL)          : ${res.expected_loss:>10,.2f}  (EL = PD × LGD × EAD)             │")
    print("├" + "─" * 76 + "┤")

    # Model & Challenger
    print("│  MODEL ARCHITECTURES:                                                      │")
    print(f"│    • Primary Regulatory Model    : {res.model:<40}│")
    print(f"│    • ML Challenger (XGBoost) PD  : {res.challenger_pd:>8.2%}                                         │")
    if res.model_comparison:
        for m in res.model_comparison.models:
            print(f"│      - {m.model_name:<34}: PD={m.predicted_pd:>6.2%} (AUC={m.auc_roc:.3f}, KS={m.ks_stat:.3f}) │")
    print("├" + "─" * 76 + "┤")

    # Risk Drivers (SHAP)
    print("│  TOP RISK DRIVERS (SHAP Game-Theoretic Attributions):                      │")
    for d in res.top_risk_drivers:
        symbol = "▲" if d.direction == "INCREASES_RISK" else "▼"
        print(f"│    {symbol} {d.feature_label:<32}: impact={d.impact:>+7.4f} ({d.direction:<14}) │")
    print("├" + "─" * 76 + "┤")

    # Macro Stress Test
    if res.stress_test_summary:
        print("│  MACROECONOMIC STRESS TESTING (IFRS 9 / CCAR Sensitivity):                │")
        for sc, stressed_loss in res.stress_test_summary.items():
            print(f"│    • {sc:<30}: Stressed EL = ${stressed_loss:>10,.2f}               │")
    print("├" + "─" * 76 + "┤")

    # Policy Decision
    decision_badge = f"[{res.policy_decision.value}]"
    print(f"│  UNDERWRITING POLICY DECISION: {decision_badge:<43}│")
    print("│  Decision Rationale:                                                       │")
    for r in res.policy_reasons:
        print(f"│    • {r:<70}│")
    print("├" + "─" * 76 + "┤")

    # Agent Narrative
    print("│  EXECUTIVE RISK MEMO (Agentic Synthesis):                                  │")
    for line in res.explanation_summary.split("\n"):
        if line.strip():
            # Wrap lines if long
            wrapped = [line[i:i + 72] for i in range(0, len(line), 72)]
            for w in wrapped:
                print(f"│    {w:<72}│")
    print("└" + "─" * 76 + "┘\n")


def main():
    parser = argparse.ArgumentParser(description="OpenClaw Credit Risk Agent CLI")
    parser.add_argument("--sample", choices=["low_risk", "borderline", "high_risk"], default="low_risk",
                        help="Run predefined sample profile")
    parser.add_argument("--json-input", type=str, default=None,
                        help="Path to JSON file containing applicant data")
    parser.add_argument("--interactive", action="store_true",
                        help="Prompt interactively for loan applicant fields")
    args = parser.parse_args()

    print_banner()

    orchestrator = get_orchestrator()

    if args.json_input:
        with open(args.json_input, "r") as f:
            data = json.load(f)
        print(f"\nEvaluating applicant from JSON file: {args.json_input}")
    elif args.interactive:
        print("\nEnter applicant details (press Enter to accept default):")
        loan_amnt = float(input("  Loan Amount [$15,000]: ") or 15000.0)
        annual_inc = float(input("  Annual Income [$75,000]: ") or 75000.0)
        int_rate = float(input("  Interest Rate (%) [11.5]: ") or 11.5)
        dti = float(input("  Debt-to-Income (%) [18.0]: ") or 18.0)
        grade = input("  Loan Grade (A-G) [B]: ").strip().upper() or "B"
        home_own = input("  Home Ownership (RENT/OWN/MORTGAGE) [MORTGAGE]: ").strip().upper() or "MORTGAGE"
        purpose = input("  Purpose [debt_consolidation]: ").strip() or "debt_consolidation"

        data = {
            "application_id": "APP-INTERACTIVE",
            "loan_amnt": loan_amnt,
            "annual_inc": annual_inc,
            "int_rate": int_rate,
            "dti": dti,
            "grade": grade,
            "home_ownership": home_own,
            "purpose": purpose,
        }
    else:
        profile_name = args.sample
        data = SAMPLE_PROFILES[profile_name]
        print(f"\nEvaluating sample profile: [{profile_name.upper()}]")

    print("\n⚡ Orchestrating multi-agent credit risk assessment...")
    assessment = orchestrator.assess_applicant(data)
    display_assessment(assessment)


if __name__ == "__main__":
    main()
