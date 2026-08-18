from setuptools import setup, find_packages

setup(
    name="openclaw_credit_risk_agent",
    version="1.0.0",
    description="Enterprise Credit Risk Assessment Multi-Agent System with Basel III Scorecard & ML Challenger Models",
    author="Pulkit Chauhan",
    packages=["openclaw_credit_risk_agent", "openclaw_credit_risk_agent.agents", "openclaw_credit_risk_agent.config", "openclaw_credit_risk_agent.tools"],
    package_dir={"openclaw_credit_risk_agent": "."},
    include_package_data=True,
    package_data={
        "openclaw_credit_risk_agent": ["data/*", "data/ml_output/*", "config/*.json"],
    },
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "joblib>=1.3.0",
        "shap>=0.43.0",
        "pydantic>=2.0.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "python-dotenv>=1.0.0",
        "groq>=0.9.0",
    ],
)
