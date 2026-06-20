import os
import re
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
# pyrefly: ignore [missing-import]
from groq import Groq

load_dotenv()

DEFAULT_MODELS = [
    "llama-3.1-8b-instant",
    "llama3-8b-8192"
]


class PolicyDeductionEngine:
    """
    Production-grade rule engine and explanation generator for HR late attendance policies.
    Handles deterministic math via code and shifts the LLM's role purely to narrative translation.
    """
    
    def __init__(self, api_key: Optional[str] = None, models: Optional[List[str]] = None):
        key = api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise ValueError("Groq API Key is missing. Ensure GROQ_API_KEY is set in environment.")
        
        self.client = Groq(api_key=key)
        self.models = models or DEFAULT_MODELS

    @staticmethod
    def select_top_chunk(chunks: List[Dict[str, Any]]) -> Optional[str]:
        """Selects the single highest scoring document chunk safely."""
        if not chunks:
            return None
        sorted_chunks = sorted(
            chunks, 
            key=lambda x: x.get("rrf_score", x.get("score", 0)), 
            reverse=True
        )
        return sorted_chunks[0].get("text", "")

    @staticmethod
    def extract_late_count(query: str) -> Optional[int]:
        """Extracts integers cleanly from input queries using regex."""
        match = re.search(r"(\d+)", query)
        return int(match.group(1)) if match else None

    @staticmethod
    def calculate_hr_deduction(late_count: int) -> Dict[str, int]:
        """
        Deterministic HR Rule Engine.
        Rule: First 4 lates = 1 day deduction. Every 3 additional lates = 1 day deduction.
        If leave balance is 0, all deductions apply directly to basic salary.
        """
        if late_count < 1:
            return {"initial_deduction": 0, "additional_deduction": 0, "total_deduction": 0}

        initial_deduction = 1 if late_count >= 4 else 0

        additional_instances = max(0, late_count - 4)
        additional_deduction = additional_instances // 3
        
        total_deduction = initial_deduction + additional_deduction
        
        return {
            "initial_deduction": initial_deduction,
            "additional_deduction": additional_deduction,
            "total_deduction": total_deduction
        }

    @staticmethod
    def _generate_prompt(query: str, context: str, late_count: int, metrics: Dict[str, int]) -> str:
        """Constructs a hyper-focused context block that forbids LLM mathematical derivation."""
        return f"""You are an HR Policy Compliance Assistant. Your single objective is to translate an established mathematical result into a clear, text-based breakdown utilizing exclusively the provided HR policy context.

CRITICAL INSTRUCTIONS:
1. Do NOT alter, calculate, recalculate, or verify any numbers provided under 'PRE-COMPUTED METRICS'.
2. Rely strictly on the facts provided below. Do not assume or extrapolate.
3. Keep the explanation professional, brief, and directly maps the pre-computed metrics to the policy clauses.

PRE-COMPUTED METRICS (TRUTH SOURCE):
- Total Late Arrivals: {late_count}
- Penalty for the First 4 Late Arrivals: {metrics['initial_deduction']} day(s) basic salary deduction (due to zero available leave balance).
- Penalty for Additional Late Arrivals: {metrics['additional_deduction']} day(s) basic salary deduction (calculated from remaining instances grouped in blocks of 3).
- Absolute Total Salary Deduction: {metrics['total_deduction']} day(s) basic salary.

ESTABLISHED COMPANY POLICY CONTEXT:
\"\"\"
{context}
\"\"\"

USER QUERY:
{query}

Provide a concise, step-by-step breakdown clarifying how the absolute total salary deduction of exactly {metrics['total_deduction']} day(s) perfectly aligns with the policy clauses. Ensure zero mathematical calculation is performed in your narrative text."""

    def _execute_groq_call(self, prompt: str) -> str:
        """Executes the chat completion loop utilizing reliable fallback architecture."""
        last_error = "No models executed."
        
        for model in self.models:
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are a rigid legal and HR verification model. You do not calculate math. You strictly justify predetermined figures using provided text clauses."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                    max_tokens=500
                )
                return response.choices[0].message.content
            except Exception as e:
                last_error = str(e)
                continue
                
        return f"System Error - Groq execution failed down the pipeline. Details: {last_error}"

    @staticmethod
    def _clean_whitespace(text: str) -> str:
        """Cleans messy spacing, tracking, and double newline artifacts for final JSON responses."""
        return re.sub(r"\s+", " ", text).strip()

    def process_pipeline(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        """Main execution gateway accepting input text matrices and mapping results."""

        context = self.select_top_chunk(chunks)
        if not context:
            return "Execution halted: Insufficient or missing company policy references provided."

        late_count = self.extract_late_count(query)
        if late_count is None:
            return "Execution halted: Failed to extract a valid numerical late arrival count from the query."


        metrics = self.calculate_hr_deduction(late_count)

        prompt = self._generate_prompt(query, context, late_count, metrics)

        raw_output = self._execute_groq_call(prompt)
        return self._clean_whitespace(raw_output)

def generate_answer(query: str, chunks: list) -> str:
    try:
        engine = PolicyDeductionEngine()
        return engine.process_pipeline(query, chunks)
    except Exception as pipeline_error:
        return f"Critical Pipeline System Error: {str(pipeline_error)}"