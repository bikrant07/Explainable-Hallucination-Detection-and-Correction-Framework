"""
flag.py
Combines multiple education-specific hallucination detection signals into a unified risk score and highlights suspicious spans.
Specialized for educational content, curriculum, and pedagogical discussions.
"""
from typing import List, Tuple, Dict
import numpy as np

def aggregate_risk(self_consistency: float, uncertainty: float, metamorphic: float, n_flagged: int, n_total: int, consensus: dict = None, openai_verdict: str = None, gemini_verdict: str = None, openai_correction: str = None, gemini_correction: str = None, user_output: str = None) -> float:
    """
    Combine detection signals, consensus, and LLM verdicts/corrections into a single hallucination risk score (0=low, 1=high).
    - If either LLM provides a non-N/A correction that differs from the user output, risk=1.0 (hallucinated)
    - If both verdicts are YES and corrections are N/A or match the user output, risk=0.1 (not hallucinated)
    - Otherwise, use original aggregation
    """
    def is_correction_flagged(correction, user_output):
        if correction and correction.strip().upper() != "N/A":
            # If correction is not the same as user output (ignoring trivial whitespace)
            return correction.strip() != (user_output or '').strip()
        return False
    # Correction-based override
    if is_correction_flagged(openai_correction, user_output) or is_correction_flagged(gemini_correction, user_output):
        return 1.0
    # LLM verdict override
    if (openai_verdict and openai_verdict.upper() == "YES") and (gemini_verdict and gemini_verdict.upper() == "YES"):
        if (not is_correction_flagged(openai_correction, user_output)) and (not is_correction_flagged(gemini_correction, user_output)):
            return 0.1
    if (openai_verdict and openai_verdict.upper() == "NO") and (gemini_verdict and gemini_verdict.upper() == "NO"):
        if consensus is not None and not consensus.get('llm_agree', True):
            return 1.0
    # Consensus-based override
    if consensus is not None:
        if not consensus.get('llm_agree', True):
            return 0.85
        elif consensus.get('llm_agree', False):
            return 0.1
    # Fallback to original aggregation
    sc = 1 - self_consistency  # low similarity = high risk
    uc = min(uncertainty / 2.0, 1.0) if uncertainty is not None else 0.5  # scale entropy
    mt = 1 - metamorphic  # low stability = high risk
    flag_frac = n_flagged / max(n_total, 1)
    risk = 0.35 * sc + 0.25 * uc + 0.2 * mt + 0.2 * flag_frac
    return float(np.clip(risk, 0, 1))

def highlight_conflicts(text: str, flagged_spans: List[Tuple[str, float]]) -> str:
    """
    Highlight flagged/conflicting spans in the text using color tags (for Streamlit rendering).
    """
    highlighted = text
    for sent, sim in flagged_spans:
        # Use red for high risk
        if sent.strip() in highlighted:
            highlighted = highlighted.replace(sent.strip(), f'<span style="background-color:#ffcccc">{sent.strip()}</span>')
    return highlighted

def explain_risk(risk: float) -> str:
    """
    Return a human-readable explanation for the risk score.
    """
    if risk < 0.2:
        return "Low risk: Output is consistent, factual, and stable."
    elif risk < 0.5:
        return "Moderate risk: Some inconsistencies or uncertainty detected."
    elif risk < 0.8:
        return "High risk: Multiple signals indicate possible hallucination."
    else:
        return "Very high risk: Output is likely hallucinated or contains factual errors." 