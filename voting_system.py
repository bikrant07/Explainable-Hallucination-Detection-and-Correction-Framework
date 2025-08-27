"""
voting_system.py
Implements a voting mechanism to combine evidence from OpenAI, Gemini, and Wikipedia.
Provides consensus-based verdict determination for educational content verification.
"""
from typing import Dict, List, Any, Tuple
import numpy as np
from enum import Enum

class Verdict(Enum):
    YES = "YES"
    NO = "NO"
    UNKNOWN = "UNKNOWN"
    ERROR = "ERROR"

class VotingSystem:
    """
    Voting system that combines evidence from multiple sources.
    """
    
    def __init__(self, weights: Dict[str, float] = None):
        """
        Initialize voting system with optional weights for different sources.
        
        Args:
            weights: Dictionary mapping source names to their weights
        """
        self.weights = weights or {
            "openai": 0.4,
            "gemini": 0.4,
            "wikipedia": 0.2
        }
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        self.weights = {k: v/total_weight for k, v in self.weights.items()}
    
    def vote(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Perform voting based on results from different sources.
        
        Args:
            results: Dictionary containing results from each source
                    Format: {"openai": {...}, "gemini": {...}, "wikipedia": {...}}
        
        Returns:
            Dictionary containing final verdict and voting details
        """
        votes = {}
        confidences = {}
        explanations = []
        
        # Process each source
        for source, result in results.items():
            if source not in self.weights:
                continue
                
            verdict = result.get("verdict", "UNKNOWN")
            confidence = result.get("confidence", 0.0)
            explanation = result.get("explanation", "")
            
            # Convert verdict to numeric score
            if verdict == "YES":
                vote_score = 1.0
            elif verdict == "NO":
                vote_score = 0.0
            elif verdict == "UNKNOWN":
                vote_score = 0.5
            else:  # ERROR or other
                vote_score = 0.5
                confidence = 0.0
            
            votes[source] = vote_score
            confidences[source] = confidence
            explanations.append(f"{source.upper()}: {verdict} (confidence: {confidence:.2f})")
        
        # Calculate weighted vote
        final_verdict, final_confidence, voting_details = self._calculate_weighted_vote(
            votes, confidences
        )
        
        # Determine consensus level
        consensus_level = self._calculate_consensus_level(votes, confidences)
        
        # Format final explanation
        final_explanation = self._format_final_explanation(
            final_verdict, final_confidence, consensus_level, explanations
        )
        
        return {
            "final_verdict": final_verdict,
            "final_confidence": final_confidence,
            "consensus_level": consensus_level,
            "explanation": final_explanation,
            "voting_details": voting_details,
            "source_results": results,
            "individual_votes": votes,
            "individual_confidences": confidences
        }
    
    def _calculate_weighted_vote(self, votes: Dict[str, float], confidences: Dict[str, float]) -> Tuple[str, float, Dict[str, Any]]:
        """
        Calculate weighted vote based on individual votes and confidences.
        """
        weighted_sum = 0.0
        total_weight = 0.0
        voting_details = {}
        
        for source, vote in votes.items():
            weight = self.weights[source]
            confidence = confidences[source]
            
            # Adjust weight by confidence
            adjusted_weight = weight * confidence
            weighted_sum += vote * adjusted_weight
            total_weight += adjusted_weight
            
            voting_details[source] = {
                "vote": vote,
                "confidence": confidence,
                "weight": weight,
                "adjusted_weight": adjusted_weight,
                "contribution": vote * adjusted_weight
            }
        
        if total_weight == 0:
            return "UNKNOWN", 0.0, voting_details
        
        # Calculate final score
        final_score = weighted_sum / total_weight
        
        # Determine verdict based on score
        if final_score > 0.7:
            final_verdict = "YES"
        elif final_score < 0.3:
            final_verdict = "NO"
        else:
            final_verdict = "UNKNOWN"
        
        # Calculate final confidence
        final_confidence = min(final_score if final_verdict == "YES" else (1 - final_score), 1.0)
        
        return final_verdict, final_confidence, voting_details
    
    def _calculate_consensus_level(self, votes: Dict[str, float], confidences: Dict[str, float]) -> str:
        """
        Calculate the level of consensus among sources.
        """
        if len(votes) < 2:
            return "SINGLE_SOURCE"
        
        # Calculate variance in votes
        vote_values = list(votes.values())
        variance = np.var(vote_values) if len(vote_values) > 1 else 0
        
        # Calculate average confidence
        avg_confidence = np.mean(list(confidences.values())) if confidences else 0
        
        # Determine consensus level
        if variance < 0.1 and avg_confidence > 0.7:
            return "HIGH_CONSENSUS"
        elif variance < 0.2 and avg_confidence > 0.5:
            return "MEDIUM_CONSENSUS"
        elif variance < 0.3:
            return "LOW_CONSENSUS"
        else:
            return "NO_CONSENSUS"
    
    def _format_final_explanation(self, verdict: str, confidence: float, consensus_level: str, explanations: List[str]) -> str:
        """
        Format the final explanation based on voting results.
        """
        explanation_parts = []
        
        # Add verdict
        if verdict == "YES":
            explanation_parts.append("✅ **FINAL VERDICT: SUPPORTED**")
        elif verdict == "NO":
            explanation_parts.append("❌ **FINAL VERDICT: NOT SUPPORTED**")
        else:
            explanation_parts.append("❓ **FINAL VERDICT: UNKNOWN**")
        
        # Add confidence
        explanation_parts.append(f"**Confidence:** {confidence:.2f}")
        
        # Add consensus level
        consensus_descriptions = {
            "HIGH_CONSENSUS": "High consensus among sources",
            "MEDIUM_CONSENSUS": "Medium consensus among sources", 
            "LOW_CONSENSUS": "Low consensus among sources",
            "NO_CONSENSUS": "No consensus among sources",
            "SINGLE_SOURCE": "Single source available"
        }
        explanation_parts.append(f"**Consensus:** {consensus_descriptions.get(consensus_level, consensus_level)}")
        
        # Add individual source explanations
        explanation_parts.append("\n**Source Results:**")
        for explanation in explanations:
            explanation_parts.append(f"- {explanation}")
        
        return "\n".join(explanation_parts)

def combine_evidence(openai_result: Dict[str, Any], 
                    gemini_result: Dict[str, Any], 
                    wikipedia_result: Dict[str, Any],
                    weights: Dict[str, float] = None) -> Dict[str, Any]:
    """
    Combine evidence from all three sources using voting system.
    
    Args:
        openai_result: Result from OpenAI verification
        gemini_result: Result from Gemini verification
        wikipedia_result: Result from Wikipedia verification
        weights: Optional weights for different sources
    
    Returns:
        Combined result with final verdict and voting details
    """
    voting_system = VotingSystem(weights)
    
    results = {
        "openai": openai_result,
        "gemini": gemini_result,
        "wikipedia": wikipedia_result
    }
    
    return voting_system.vote(results)

def format_voting_result(result: Dict[str, Any]) -> str:
    """
    Format voting result for display.
    """
    verdict = result.get("final_verdict", "UNKNOWN")
    confidence = result.get("final_confidence", 0.0)
    consensus_level = result.get("consensus_level", "UNKNOWN")
    explanation = result.get("explanation", "")
    
    # Color coding based on verdict - using lighter, more readable colors
    if verdict == "YES":
        bg_color = "#e8f5e8"  # Light green background
        text_color = "#2d5a2d"  # Dark green text
        status = "✅ SUPPORTED"
    elif verdict == "NO":
        bg_color = "#ffe8e8"  # Light red background
        text_color = "#5a2d2d"  # Dark red text
        status = "❌ NOT SUPPORTED"
    else:
        bg_color = "#fff3e0"  # Light orange background
        text_color = "#5a3d2d"  # Dark orange text
        status = "❓ UNKNOWN"
    
    # Consensus level indicator
    consensus_indicators = {
        "HIGH_CONSENSUS": "🟢",
        "MEDIUM_CONSENSUS": "🟡", 
        "LOW_CONSENSUS": "🟠",
        "NO_CONSENSUS": "🔴",
        "SINGLE_SOURCE": "⚪"
    }
    
    consensus_indicator = consensus_indicators.get(consensus_level, "⚪")
    
    output = f"""
<div style='padding:1em;background-color:{bg_color};border-radius:10px;margin-bottom:1em;border-left:4px solid {text_color};'>
<h3 style='margin-bottom:0.5em;color:{text_color};'>{status} {consensus_indicator}</h3>
<b style='color:{text_color};'>Confidence:</b> <span style='color:{text_color};'>{confidence:.2f}</span><br>
<b style='color:{text_color};'>Consensus Level:</b> <span style='color:{text_color};'>{consensus_level}</span><br>
<b style='color:{text_color};'>Explanation:</b> <span style='color:{text_color};'>{explanation}</span>
</div>
"""
    
    return output

def get_voting_summary(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get a summary of the voting results.
    """
    voting_details = result.get("voting_details", {})
    individual_votes = result.get("individual_votes", {})
    individual_confidences = result.get("individual_confidences", {})
    
    summary = {
        "final_verdict": result.get("final_verdict"),
        "final_confidence": result.get("final_confidence"),
        "consensus_level": result.get("consensus_level"),
        "source_summary": {}
    }
    
    for source in ["openai", "gemini", "wikipedia"]:
        if source in voting_details:
            details = voting_details[source]
            summary["source_summary"][source] = {
                "vote": "YES" if details["vote"] > 0.7 else "NO" if details["vote"] < 0.3 else "UNKNOWN",
                "confidence": details["confidence"],
                "weight": details["weight"],
                "contribution": details["contribution"]
            }
    
    return summary

# Example usage
if __name__ == "__main__":
    # Example results from different sources
    openai_result = {
        "verdict": "YES",
        "confidence": 0.8,
        "explanation": "OpenAI supports this claim based on educational knowledge."
    }
    
    gemini_result = {
        "verdict": "YES", 
        "confidence": 0.7,
        "explanation": "Gemini also supports this claim."
    }
    
    wikipedia_result = {
        "verdict": "UNKNOWN",
        "confidence": 0.5,
        "explanation": "Wikipedia has limited information on this topic."
    }
    
    # Combine evidence
    combined_result = combine_evidence(openai_result, gemini_result, wikipedia_result)
    
    print("Voting Result:")
    print(combined_result["final_verdict"])
    print(f"Confidence: {combined_result['final_confidence']:.2f}")
    print(f"Consensus: {combined_result['consensus_level']}")
    print("\nExplanation:")
    print(combined_result["explanation"]) 