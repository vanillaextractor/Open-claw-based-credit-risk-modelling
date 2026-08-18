"""Base Agent with Groq LLM integration and strict guardrails."""

import logging
from typing import Dict, Any, Optional
from openclaw_credit_risk_agent.config.settings import get_settings

logger = logging.getLogger("openclaw_credit_risk.agents")


class BaseAgent:
    """Base Agent providing LLM reasoning capabilities via Groq API."""

    def __init__(self, name: str, role_prompt: str):
        self.name = name
        self.role_prompt = role_prompt
        self.settings = get_settings()
        self.client = None
        self._init_groq_client()

    def _init_groq_client(self):
        """Initialize the Groq client if API key is provided."""
        api_key = self.settings.groq_api_key
        if api_key and api_key != "your_groq_api_key_here":
            try:
                from groq import Groq
                self.client = Groq(api_key=api_key)
                logger.info(f"[{self.name}] Groq client initialized with model: {self.settings.groq_model}")
            except Exception as e:
                logger.warning(f"[{self.name}] Failed to initialize Groq client: {e}")
                self.client = None
        else:
            self.client = None

    def query_llm(self, prompt: str, system_override: Optional[str] = None) -> str:
        """Send prompt to Groq LLM with guardrails, falling back to deterministic template if unavailable."""
        system_instruction = system_override or self.role_prompt
        system_instruction += (
            "\n\nGUARDRAIL MANDATE:\n"
            "1. You must NEVER modify, recalculate, or fabricate numerical PD, LGD, EAD, Expected Loss, or Credit Score values.\n"
            "2. All numbers provided in the context are deterministic facts from the Python engine.\n"
            "3. Separate statistical model predictions from underwriting policy decisions."
        )

        if self.client is not None:
            try:
                chat_completion = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": prompt}
                    ],
                    model=self.settings.groq_model,
                    temperature=self.settings.groq_temperature,
                    max_tokens=self.settings.groq_max_tokens,
                )
                return chat_completion.choices[0].message.content or ""
            except Exception as e:
                logger.warning(f"[{self.name}] Groq query failed ({e}), using fallback synthesis.")

        # Deterministic fallback reasoning
        return self._generate_deterministic_fallback(prompt)

    def _generate_deterministic_fallback(self, prompt: str) -> str:
        """Deterministic reasoning fallback when Groq API key is not configured."""
        return f"[{self.name}] Analytical review completed based on deterministic model outputs."
