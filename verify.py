"""
verify.py
Implements education-specific claim verification using OpenAI API against retrieved educational facts.
Specialized for verifying educational content, curriculum statements, and pedagogical claims.
"""
import os
from typing import Dict, List, Tuple
import re

def verify_claim_with_openai(claim: str, facts: List[str], api_key: str, model: str = "gpt-3.5-turbo") -> Dict:
    """
    Verify a claim against retrieved facts using OpenAI API.
    Returns structured verification result.
    """
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
        
        # Format facts for the prompt
        facts_text = "\n".join([f"- {fact}" for fact in facts])
        
        prompt = f"""
You are an education fact-checking expert. Given an educational claim and supporting facts, determine if the claim is supported by the facts.

CLAIM: {claim}

SUPPORTING FACTS:
{facts_text}

Please provide your analysis in the following format:

VERDICT: YES/NO
EXPLANATION: [Detailed explanation of why the claim is supported or not supported by the facts, focusing on educational accuracy]
CORRECTED_CLAIM: [If the claim is incorrect, provide a corrected version based on the facts. If correct, write "N/A"]

Focus on educational accuracy, pedagogical correctness, and curriculum alignment. Consider educational theories, teaching methods, and academic standards.
"""
        
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=500,
        )
        
        result_text = response.choices[0].message.content.strip()
        return parse_verification_response(result_text)
        
    except Exception as e:
        return {
            "verdict": "ERROR",
            "explanation": f"Error during verification: {str(e)}",
            "corrected_claim": "N/A"
        }

def verify_claim_with_gemini(claim: str, facts: List[str], api_key: str) -> Dict:
    """
    Verify a claim against retrieved facts using Gemini API (gemini-2.0-flash).
    Returns structured verification result.
    """
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        facts_text = "\n".join([f"- {fact}" for fact in facts])
        prompt = f"""
You are an education fact-checking expert. Given an educational claim and supporting facts, determine if the claim is supported by the facts.

CLAIM: {claim}

SUPPORTING FACTS:
{facts_text}

Please provide your analysis in the following format:

VERDICT: YES/NO
EXPLANATION: [Detailed explanation of why the claim is supported or not supported by the facts, focusing on educational accuracy]
CORRECTED_CLAIM: [If the claim is incorrect, provide a corrected version based on the facts. If correct, write \"N/A\"]

Focus on educational accuracy, pedagogical correctness, and curriculum alignment. Consider educational theories, teaching methods, and academic standards.
"""
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt)
        result_text = response.text.strip()
        return parse_verification_response(result_text)
    except Exception as e:
        return {
            "verdict": "ERROR",
            "explanation": f"Gemini error: {str(e)}",
            "corrected_claim": "N/A"
        }

def parse_verification_response(response: str) -> Dict:
    """
    Parse the structured verification response from OpenAI.
    """
    result = {
        "verdict": "UNKNOWN",
        "explanation": "Could not parse response",
        "corrected_claim": "N/A"
    }
    
    # Extract VERDICT
    verdict_match = re.search(r'VERDICT:\s*(YES|NO)', response, re.IGNORECASE)
    if verdict_match:
        result["verdict"] = verdict_match.group(1).upper()
    
    # Extract EXPLANATION
    explanation_match = re.search(r'EXPLANATION:\s*(.*?)(?=\n[A-Z]+:|$)', response, re.DOTALL | re.IGNORECASE)
    if explanation_match:
        result["explanation"] = explanation_match.group(1).strip()
    
    # Extract CORRECTED_CLAIM
    corrected_match = re.search(r'CORRECTED_CLAIM:\s*(.*?)(?=\n[A-Z]+:|$)', response, re.DOTALL | re.IGNORECASE)
    if corrected_match:
        corrected = corrected_match.group(1).strip()
        if corrected.upper() != "N/A":
            result["corrected_claim"] = corrected
    
    return result

def format_verification_result(result: Dict, provider: str = "OpenAI") -> str:
    """
    Format verification result for display.
    """
    verdict = result.get("verdict", "UNKNOWN")
    explanation = result.get("explanation", "")
    corrected = result.get("corrected_claim", "N/A")
    
    if verdict == "YES":
        status = f"✅ SUPPORTED by {provider}"
    elif verdict == "NO":
        status = f"❌ NOT SUPPORTED by {provider}"
    else:
        status = f"❓ UNKNOWN ({provider})"
    
    output = f"""
**{status}**

**Explanation:** {explanation}

"""
    
    if corrected != "N/A":
        output += f"**Corrected Claim:** {corrected}"
    
    return output

# Example usage
if __name__ == "__main__":
    # Test with sample data
    claim = "Paris is the capital of France and Berlin is the capital of Spain."
    facts = [
        "Paris is the capital of France.",
        "Berlin is the capital of Germany, not Spain.",
        "Madrid is the capital of Spain."
    ]
    
    # This would require an actual API key
    # result = verify_claim_with_openai(claim, facts, "your-api-key")
    # print(format_verification_result(result)) 