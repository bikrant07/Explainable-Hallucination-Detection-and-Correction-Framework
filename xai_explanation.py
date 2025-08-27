"""
xai_explanation.py
XAI (Explainable AI) module that provides comprehensive, step-by-step explanations
of the hallucination detection and correction process with visualizations.
"""
from typing import Dict, List, Any, Tuple
import difflib
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import streamlit as st

class XAIExplanation:
    """
    Provides comprehensive explanations for hallucination detection and correction with visualizations.
    """
    
    def __init__(self):
        self.explanation_steps = []
        # Set style for matplotlib
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def generate_complete_explanation(self, 
                                    original_output: str,
                                    query: str,
                                    detection_results: Dict,
                                    correction_results: Dict,
                                    evidence_set: List[Dict],
                                    voting_results: Dict) -> Dict[str, Any]:
        """
        Generate a complete, step-by-step explanation of the entire process with visualizations.
        
        Args:
            original_output: The original LLM output to be analyzed
            query: The original query/prompt
            detection_results: Results from detection algorithms
            correction_results: Results from correction process
            evidence_set: Retrieved evidence from various sources
            voting_results: Final voting and consensus results
        
        Returns:
            Complete explanation with all components and visualizations
        """
        
        explanation = {
            "summary": self._generate_summary(voting_results, detection_results),
            "step_by_step": self._generate_step_by_step_explanation(
                original_output, query, detection_results, correction_results, 
                evidence_set, voting_results
            ),
            "evidence_attribution": self._generate_evidence_attribution(evidence_set),
            "correction_explanation": self._generate_correction_explanation(
                original_output, correction_results
            ),
            "decision_rationale": self._generate_decision_rationale(voting_results),
            "risk_assessment": self._generate_risk_assessment(detection_results, voting_results),
            "visual_highlights": self._generate_visual_highlights(
                original_output, detection_results, correction_results
            ),
            "confidence_breakdown": self._generate_confidence_breakdown(voting_results),
            "recommendations": self._generate_recommendations(voting_results, detection_results),
            "visualizations": self._generate_all_visualizations(
                detection_results, correction_results, evidence_set, voting_results
            )
        }
        
        return explanation
    
    def _generate_summary(self, voting_results: Dict, detection_results: Dict) -> str:
        """Generate a high-level summary of the analysis."""
        final_verdict = voting_results.get("final_verdict", "UNKNOWN")
        final_confidence = voting_results.get("final_confidence", 0.0)
        consensus_level = voting_results.get("consensus_level", "UNKNOWN")
        
        risk_score = detection_results.get("risk_score", 0.0)
        
        summary = f"""
## 🎯 Analysis Summary

**Final Verdict**: {final_verdict}  
**Confidence**: {final_confidence:.2f}  
**Consensus Level**: {consensus_level}  
**Risk Score**: {risk_score:.2f}

### Key Findings:
"""
        
        if final_verdict == "YES":
            summary += "✅ The output appears to be **factually accurate** and well-supported by evidence."
        elif final_verdict == "NO":
            summary += "❌ The output contains **factual errors** or hallucinations that have been identified and corrected."
        else:
            summary += "❓ The output's accuracy is **uncertain** due to insufficient or conflicting evidence."
        
        return summary
    
    def _generate_step_by_step_explanation(self, 
                                         original_output: str,
                                         query: str,
                                         detection_results: Dict,
                                         correction_results: Dict,
                                         evidence_set: List[Dict],
                                         voting_results: Dict) -> List[Dict]:
        """Generate a detailed step-by-step explanation of the entire process."""
        
        steps = []
        
        # Step 1: Evidence Collection
        steps.append({
            "step": 1,
            "title": "Evidence Collection",
            "description": "Gathered supporting evidence from multiple sources",
            "details": {
                "static_kb_sources": len([e for e in evidence_set if e.get("source") == "knowledge_base"]),
                "wikipedia_sources": len([e for e in evidence_set if e.get("source") == "wikipedia"]),
                "total_sources": len(evidence_set),
                "top_evidence": evidence_set[:3] if evidence_set else []
            },
            "icon": "🔍"
        })
        
        # Step 2: Detection Analysis
        detection_signals = detection_results.get("signals", {})
        steps.append({
            "step": 2,
            "title": "Detection Analysis",
            "description": "Applied multiple detection algorithms to identify potential issues",
            "details": {
                "self_consistency": detection_signals.get("consistency_score", 0.0),
                "metamorphic_stability": detection_signals.get("stability_score", 0.0),
                "uncertainty": detection_signals.get("uncertainty_score", 0.0),
                "flagged_spans": len(detection_results.get("flagged_spans", []))
            },
            "icon": "🔬"
        })
        
        # Step 3: Multi-LLM Verification
        steps.append({
            "step": 3,
            "title": "Multi-LLM Verification",
            "description": "Verified claims using multiple AI models with structured prompts",
            "details": {
                "openai_verdict": correction_results.get("openai", {}).get("verdict", "UNKNOWN"),
                "gemini_verdict": correction_results.get("gemini", {}).get("verdict", "UNKNOWN"),
                "wikipedia_verdict": correction_results.get("wikipedia", {}).get("verdict", "UNKNOWN")
            },
            "icon": "🤖"
        })
        
        # Step 4: Evidence Fusion
        steps.append({
            "step": 4,
            "title": "Evidence Fusion",
            "description": "Combined evidence using weighted voting system",
            "details": {
                "final_verdict": voting_results.get("final_verdict", "UNKNOWN"),
                "consensus_level": voting_results.get("consensus_level", "UNKNOWN"),
                "voting_weights": voting_results.get("weights", {})
            },
            "icon": "🗳️"
        })
        
        # Step 5: Correction Generation
        if correction_results.get("corrected_claim"):
            steps.append({
                "step": 5,
                "title": "Correction Generation",
                "description": "Generated corrected version based on evidence",
                "details": {
                    "has_correction": True,
                    "correction_source": correction_results.get("correction_source", "unknown")
                },
                "icon": "✏️"
            })
        
        return steps
    
    def _generate_evidence_attribution(self, evidence_set: List[Dict]) -> Dict[str, Any]:
        """Generate evidence attribution with source credibility."""
        attribution = {
            "total_sources": len(evidence_set),
            "source_breakdown": {},
            "top_supporting": [],
            "top_contradicting": [],
            "credibility_scores": {}
        }
        
        # Count sources by type
        for evidence in evidence_set:
            source_type = evidence.get("source", "unknown")
            attribution["source_breakdown"][source_type] = attribution["source_breakdown"].get(source_type, 0) + 1
        
        # Find top supporting and contradicting evidence
        for evidence in evidence_set:
            support_score = evidence.get("support_score", 0.0)
            if support_score > 0.7:
                attribution["top_supporting"].append(evidence)
            elif support_score < 0.3:
                attribution["top_contradicting"].append(evidence)
        
        # Calculate credibility scores
        for evidence in evidence_set:
            source = evidence.get("source", "unknown")
            if source not in attribution["credibility_scores"]:
                attribution["credibility_scores"][source] = {
                    "count": 0,
                    "avg_relevance": 0.0,
                    "avg_support": 0.0
                }
            
            cred = attribution["credibility_scores"][source]
            cred["count"] += 1
            cred["avg_relevance"] += evidence.get("relevance_score", 0.0)
            cred["avg_support"] += evidence.get("support_score", 0.0)
        
        # Calculate averages
        for source in attribution["credibility_scores"]:
            cred = attribution["credibility_scores"][source]
            if cred["count"] > 0:
                cred["avg_relevance"] /= cred["count"]
                cred["avg_support"] /= cred["count"]
        
        return attribution
    
    def _generate_correction_explanation(self, original_output: str, correction_results: Dict) -> Dict[str, Any]:
        """Generate detailed explanation of corrections made."""
        correction_explanation = {
            "has_corrections": False,
            "corrections": [],
            "diff_analysis": {},
            "rationale": {}
        }
        
        # Check for corrections from different sources
        sources = ["openai", "gemini", "wikipedia"]
        
        for source in sources:
            source_result = correction_results.get(source, {})
            corrected_claim = source_result.get("corrected_claim")
            
            if corrected_claim and corrected_claim != "N/A":
                correction_explanation["has_corrections"] = True
                
                # Generate diff
                diff = list(difflib.ndiff(original_output.split(), corrected_claim.split()))
                
                # Analyze changes
                additions = [word[2:] for word in diff if word.startswith('+ ')]
                deletions = [word[2:] for word in diff if word.startswith('- ')]
                
                correction_info = {
                    "source": source.upper(),
                    "original": original_output,
                    "corrected": corrected_claim,
                    "additions": additions,
                    "deletions": deletions,
                    "rationale": source_result.get("explanation", ""),
                    "confidence": source_result.get("confidence", 0.0)
                }
                
                correction_explanation["corrections"].append(correction_info)
        
        return correction_explanation
    
    def _generate_decision_rationale(self, voting_results: Dict) -> Dict[str, Any]:
        """Generate rationale for the final decision."""
        rationale = {
            "final_verdict": voting_results.get("final_verdict", "UNKNOWN"),
            "final_confidence": voting_results.get("final_confidence", 0.0),
            "consensus_level": voting_results.get("consensus_level", "UNKNOWN"),
            "source_contributions": {},
            "decision_factors": []
        }
        
        # Analyze source contributions
        voting_details = voting_results.get("voting_details", {})
        for source, details in voting_details.items():
            rationale["source_contributions"][source] = {
                "vote": details.get("vote", 0.0),
                "confidence": details.get("confidence", 0.0),
                "weight": details.get("weight", 0.0),
                "contribution": details.get("contribution", 0.0)
            }
        
        # Identify decision factors
        if rationale["consensus_level"] == "HIGH_CONSENSUS":
            rationale["decision_factors"].append("High agreement among all sources")
        elif rationale["consensus_level"] == "MEDIUM_CONSENSUS":
            rationale["decision_factors"].append("Moderate agreement with some disagreement")
        elif rationale["consensus_level"] == "LOW_CONSENSUS":
            rationale["decision_factors"].append("Low agreement among sources")
        else:
            rationale["decision_factors"].append("No clear consensus among sources")
        
        if rationale["final_confidence"] > 0.8:
            rationale["decision_factors"].append("High confidence in the decision")
        elif rationale["final_confidence"] < 0.5:
            rationale["decision_factors"].append("Low confidence due to insufficient evidence")
        
        return rationale
    
    def _generate_risk_assessment(self, detection_results: Dict, voting_results: Dict) -> Dict[str, Any]:
        """Generate comprehensive risk assessment."""
        risk_score = detection_results.get("risk_score", 0.0)
        final_verdict = voting_results.get("final_verdict", "UNKNOWN")
        
        risk_assessment = {
            "overall_risk": risk_score,
            "risk_level": self._categorize_risk(risk_score),
            "risk_factors": [],
            "mitigation_suggestions": []
        }
        
        # Identify risk factors
        detection_signals = detection_results.get("signals", {})
        
        if detection_signals.get("consistency_score", 1.0) < 0.7:
            risk_assessment["risk_factors"].append("Low self-consistency across multiple generations")
        
        if detection_signals.get("stability_score", 1.0) < 0.7:
            risk_assessment["risk_factors"].append("Unstable responses to paraphrased queries")
        
        if detection_signals.get("uncertainty_score", 0.0) > 0.5:
            risk_assessment["risk_factors"].append("High uncertainty in model predictions")
        
        if final_verdict == "NO":
            risk_assessment["risk_factors"].append("Factual errors detected by verification models")
        
        # Generate mitigation suggestions
        if risk_score > 0.7:
            risk_assessment["mitigation_suggestions"].extend([
                "Verify claims against authoritative sources",
                "Cross-check with multiple independent sources",
                "Consider using more conservative language",
                "Provide citations for factual statements"
            ])
        
        return risk_assessment
    
    def _generate_visual_highlights(self, original_output: str, detection_results: Dict, correction_results: Dict) -> Dict[str, Any]:
        """Generate visual highlighting for the output."""
        highlights = {
            "original_text": original_output,
            "highlighted_spans": [],
            "correction_highlights": [],
            "confidence_indicators": []
        }
        
        # Highlight flagged spans
        flagged_spans = detection_results.get("flagged_spans", [])
        for span, score in flagged_spans:
            highlights["highlighted_spans"].append({
                "text": span,
                "score": score,
                "color": "red" if score > 0.7 else "orange" if score > 0.4 else "yellow"
            })
        
        # Highlight corrections
        for source in ["openai", "gemini"]:
            source_result = correction_results.get(source, {})
            corrected_claim = source_result.get("corrected_claim")
            
            if corrected_claim and corrected_claim != "N/A":
                diff = list(difflib.ndiff(original_output.split(), corrected_claim.split()))
                
                for word in diff:
                    if word.startswith('+ '):
                        highlights["correction_highlights"].append({
                            "text": word[2:],
                            "type": "addition",
                            "source": source.upper()
                        })
                    elif word.startswith('- '):
                        highlights["correction_highlights"].append({
                            "text": word[2:],
                            "type": "deletion",
                            "source": source.upper()
                        })
        
        return highlights
    
    def _generate_confidence_breakdown(self, voting_results: Dict) -> Dict[str, Any]:
        """Generate detailed confidence breakdown."""
        confidence_breakdown = {
            "overall_confidence": voting_results.get("final_confidence", 0.0),
            "source_confidences": {},
            "confidence_factors": []
        }
        
        voting_details = voting_results.get("voting_details", {})
        for source, details in voting_details.items():
            confidence_breakdown["source_confidences"][source] = {
                "confidence": details.get("confidence", 0.0),
                "weight": details.get("weight", 0.0),
                "contribution": details.get("contribution", 0.0)
            }
        
        # Identify confidence factors
        consensus_level = voting_results.get("consensus_level", "UNKNOWN")
        if consensus_level == "HIGH_CONSENSUS":
            confidence_breakdown["confidence_factors"].append("High agreement among sources increases confidence")
        elif consensus_level == "NO_CONSENSUS":
            confidence_breakdown["confidence_factors"].append("Disagreement among sources reduces confidence")
        
        return confidence_breakdown
    
    def _generate_recommendations(self, voting_results: Dict, detection_results: Dict) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        final_verdict = voting_results.get("final_verdict", "UNKNOWN")
        risk_score = detection_results.get("risk_score", 0.0)
        
        if final_verdict == "NO":
            recommendations.extend([
                "Review and correct the identified factual errors",
                "Verify information against authoritative sources",
                "Consider providing citations for factual claims",
                "Use more conservative language when uncertain"
            ])
        
        if risk_score > 0.7:
            recommendations.extend([
                "Implement additional fact-checking procedures",
                "Cross-verify with multiple independent sources",
                "Consider using domain-specific knowledge bases",
                "Provide uncertainty estimates for claims"
            ])
        
        if voting_results.get("consensus_level") == "LOW_CONSENSUS":
            recommendations.append("Seek additional evidence to resolve conflicting information")
        
        return recommendations
    
    def _categorize_risk(self, risk_score: float) -> str:
        """Categorize risk level based on score."""
        if risk_score < 0.2:
            return "LOW"
        elif risk_score < 0.5:
            return "MODERATE"
        elif risk_score < 0.8:
            return "HIGH"
        else:
            return "VERY HIGH"
    
    def format_explanation_for_display(self, explanation: Dict[str, Any]) -> str:
        """Format the complete explanation for display in the UI."""
        html = f"""
        <div style="font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto;">
            {explanation['summary']}
            
            <h2>🔍 Step-by-Step Analysis</h2>
        """
        
        for step in explanation['step_by_step']:
            html += f"""
            <div style="border-left: 4px solid #007bff; padding-left: 15px; margin: 10px 0;">
                <h3>{step['icon']} Step {step['step']}: {step['title']}</h3>
                <p>{step['description']}</p>
                <ul>
            """
            
            for key, value in step['details'].items():
                if isinstance(value, float):
                    html += f"<li><strong>{key.replace('_', ' ').title()}:</strong> {value:.2f}</li>"
                else:
                    html += f"<li><strong>{key.replace('_', ' ').title()}:</strong> {value}</li>"
            
            html += "</ul></div>"
        
        # Add evidence attribution
        if explanation['evidence_attribution']['total_sources'] > 0:
            html += f"""
            <h2>📚 Evidence Attribution</h2>
            <p><strong>Total Sources:</strong> {explanation['evidence_attribution']['total_sources']}</p>
            <ul>
            """
            
            for source, count in explanation['evidence_attribution']['source_breakdown'].items():
                html += f"<li><strong>{source.title()}:</strong> {count} sources</li>"
            
            html += "</ul>"
        
        # Add correction explanation
        if explanation['correction_explanation']['has_corrections']:
            html += """
            <h2>✏️ Corrections Made</h2>
            """
            
            for correction in explanation['correction_explanation']['corrections']:
                html += f"""
                <div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin: 10px 0;">
                    <h4>Correction by {correction['source']}</h4>
                    <p><strong>Original:</strong> {correction['original']}</p>
                    <p><strong>Corrected:</strong> {correction['corrected']}</p>
                    <p><strong>Rationale:</strong> {correction['rationale']}</p>
                </div>
                """
        
        # Add risk assessment
        risk = explanation['risk_assessment']
        html += f"""
        <h2>⚠️ Risk Assessment</h2>
        <p><strong>Risk Level:</strong> {risk['risk_level']} ({risk['overall_risk']:.2f})</p>
        <h4>Risk Factors:</h4>
        <ul>
        """
        
        for factor in risk['risk_factors']:
            html += f"<li>{factor}</li>"
        
        html += "</ul>"
        
        # Add recommendations
        if explanation['recommendations']:
            html += """
            <h2>💡 Recommendations</h2>
            <ul>
            """
            
            for rec in explanation['recommendations']:
                html += f"<li>{rec}</li>"
            
            html += "</ul>"
        
        html += "</div>"
        
        return html 

    def _generate_all_visualizations(self, detection_results: Dict, correction_results: Dict, 
                                   evidence_set: List[Dict], voting_results: Dict) -> Dict[str, Any]:
        """Generate essential visualizations for the explanation."""
        visualizations = {
            "confidence_gauge": self._create_confidence_gauge(voting_results),
            "source_contribution_chart": self._create_source_contribution_chart(voting_results),
            "detection_signals_radar": self._create_detection_signals_radar(detection_results),
            "risk_thermometer": self._create_risk_thermometer(detection_results),
            "consensus_flowchart": self._create_consensus_flowchart(voting_results)
        }
        return visualizations
    
    def _create_confidence_gauge(self, voting_results: Dict) -> go.Figure:
        """Create a gauge chart showing final confidence."""
        confidence = voting_results.get("final_confidence", 0.0)
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = confidence * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Final Confidence Score"},
            delta = {'reference': 50},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgray"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(
            title="Confidence Assessment",
            font={'size': 16},
            height=300
        )
        
        return fig
    
    def _create_source_contribution_chart(self, voting_results: Dict) -> go.Figure:
        """Create a bar chart showing source contributions."""
        voting_details = voting_results.get("voting_details", {})
        
        sources = []
        votes = []
        confidences = []
        weights = []
        contributions = []
        
        for source, details in voting_details.items():
            sources.append(source.upper())
            votes.append(details.get("vote", 0.0))
            confidences.append(details.get("confidence", 0.0))
            weights.append(details.get("weight", 0.0))
            contributions.append(details.get("contribution", 0.0))
        
        fig = go.Figure()
        
        # Add bars for each metric
        fig.add_trace(go.Bar(
            name='Vote Score',
            x=sources,
            y=votes,
            marker_color='lightblue',
            text=[f'{v:.2f}' for v in votes],
            textposition='auto'
        ))
        
        fig.add_trace(go.Bar(
            name='Confidence',
            x=sources,
            y=confidences,
            marker_color='lightgreen',
            text=[f'{c:.2f}' for c in confidences],
            textposition='auto'
        ))
        
        fig.add_trace(go.Bar(
            name='Weight',
            x=sources,
            y=weights,
            marker_color='lightcoral',
            text=[f'{w:.2f}' for w in weights],
            textposition='auto'
        ))
        
        fig.update_layout(
            title="Source Contributions to Final Decision",
            xaxis_title="Sources",
            yaxis_title="Scores",
            barmode='group',
            height=400
        )
        
        return fig
    
    def _create_detection_signals_radar(self, detection_results: Dict) -> go.Figure:
        """Create a radar chart showing detection signals."""
        signals = detection_results.get("signals", {})
        
        categories = ['Self-Consistency', 'Metamorphic Stability', 'Uncertainty', 'Risk Score']
        values = [
            signals.get("consistency_score", 0.0),
            signals.get("stability_score", 0.0),
            1 - signals.get("uncertainty_score", 0.0),  # Invert uncertainty for positive scale
            1 - detection_results.get("risk_score", 0.0)  # Invert risk for positive scale
        ]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Detection Signals',
            line_color='blue',
            fillcolor='rgba(0, 100, 255, 0.3)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=False,
            title="Detection Signals Radar Chart",
            height=400
        )
        
        return fig
    
    def _create_risk_thermometer(self, detection_results: Dict) -> go.Figure:
        """Create a thermometer-style risk indicator."""
        risk_score = detection_results.get("risk_score", 0.0)
        
        # Determine risk level and color
        if risk_score < 0.3:
            risk_level = "LOW"
            color = "green"
        elif risk_score < 0.7:
            risk_level = "MEDIUM"
            color = "orange"
        else:
            risk_level = "HIGH"
            color = "red"
        
        fig = go.Figure()
        
        # Create thermometer
        fig.add_trace(go.Indicator(
            mode = "gauge+number",
            value = risk_score * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"Risk Level: {risk_level}"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))
        
        fig.update_layout(
            title="Risk Assessment Thermometer",
            font={'size': 16},
            height=300
        )
        
        return fig
    
    def _create_consensus_flowchart(self, voting_results: Dict) -> go.Figure:
        """Create a flowchart showing the consensus decision process."""
        consensus_level = voting_results.get("consensus_level", "UNKNOWN")
        final_verdict = voting_results.get("final_verdict", "UNKNOWN")
        
        # Define flowchart nodes and edges
        nodes = [
            {"id": "start", "label": "Start", "x": 0, "y": 0},
            {"id": "evidence", "label": "Evidence Collection", "x": 0, "y": -1},
            {"id": "verification", "label": "Multi-LLM Verification", "x": 0, "y": -2},
            {"id": "voting", "label": "Voting System", "x": 0, "y": -3},
            {"id": "consensus", "label": f"Consensus: {consensus_level}", "x": 0, "y": -4},
            {"id": "result", "label": f"Final: {final_verdict}", "x": 0, "y": -5}
        ]
        
        edges = [
            ("start", "evidence"),
            ("evidence", "verification"),
            ("verification", "voting"),
            ("voting", "consensus"),
            ("consensus", "result")
        ]
        
        # Create flowchart
        fig = go.Figure()
        
        # Add nodes
        for node in nodes:
            fig.add_trace(go.Scatter(
                x=[node["x"]],
                y=[node["y"]],
                mode='markers+text',
                marker=dict(size=20, color='lightblue'),
                text=[node["label"]],
                textposition="middle center",
                showlegend=False
            ))
        
        # Add edges
        for edge in edges:
            start_node = next(n for n in nodes if n["id"] == edge[0])
            end_node = next(n for n in nodes if n["id"] == edge[1])
            
            fig.add_trace(go.Scatter(
                x=[start_node["x"], end_node["x"]],
                y=[start_node["y"], end_node["y"]],
                mode='lines',
                line=dict(color='gray', width=2),
                showlegend=False
            ))
        
        fig.update_layout(
            title="Decision Process Flowchart",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=500
        )
        
        return fig

    def _create_evidence_relevance_heatmap(self, evidence_set: List[Dict]) -> go.Figure:
        """Create a heatmap showing evidence relevance and support scores."""
        if not evidence_set:
            return go.Figure()
        
        # Prepare data
        sources = [e.get("source", "unknown") for e in evidence_set]
        relevance_scores = [e.get("relevance_score", 0.0) for e in evidence_set]
        support_scores = [e.get("support_score", 0.0) for e in evidence_set]
        
        # Create heatmap data
        heatmap_data = np.array([relevance_scores, support_scores])
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=sources,
            y=['Relevance', 'Support'],
            colorscale='Viridis',
            text=[[f'{r:.2f}' for r in relevance_scores], 
                  [f'{s:.2f}' for s in support_scores]],
            texttemplate="%{text}",
            textfont={"size": 12},
            colorbar=dict(title="Score")
        ))
        
        fig.update_layout(
            title="Evidence Relevance and Support Heatmap",
            xaxis_title="Evidence Sources",
            yaxis_title="Metrics",
            height=300
        )
        
        return fig
    
    def _create_correction_comparison(self, correction_results: Dict) -> go.Figure:
        """Create a comparison chart of corrections from different sources."""
        sources = []
        confidences = []
        has_corrections = []
        
        for source in ["openai", "gemini", "wikipedia"]:
            source_result = correction_results.get(source, {})
            sources.append(source.upper())
            confidences.append(source_result.get("confidence", 0.0))
            has_corrections.append(1 if source_result.get("corrected_claim") and 
                                 source_result.get("corrected_claim") != "N/A" else 0)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Confidence',
            x=sources,
            y=confidences,
            marker_color='lightblue',
            text=[f'{c:.2f}' for c in confidences],
            textposition='auto'
        ))
        
        fig.add_trace(go.Bar(
            name='Has Correction',
            x=sources,
            y=has_corrections,
            marker_color='orange',
            text=['Yes' if h else 'No' for h in has_corrections],
            textposition='auto'
        ))
        
        fig.update_layout(
            title="Correction Analysis by Source",
            xaxis_title="Sources",
            yaxis_title="Scores",
            barmode='group',
            height=400
        )
        
        return fig
    
    def _create_evidence_timeline(self, evidence_set: List[Dict]) -> go.Figure:
        """Create a timeline visualization of evidence sources."""
        if not evidence_set:
            return go.Figure()
        
        # Group evidence by source
        source_counts = {}
        for evidence in evidence_set:
            source = evidence.get("source", "unknown")
            source_counts[source] = source_counts.get(source, 0) + 1
        
        sources = list(source_counts.keys())
        counts = list(source_counts.values())
        
        fig = go.Figure(data=[
            go.Bar(
                x=sources,
                y=counts,
                marker_color=['lightblue', 'lightgreen', 'lightcoral', 'lightyellow'][:len(sources)],
                text=counts,
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title="Evidence Source Distribution",
            xaxis_title="Evidence Sources",
            yaxis_title="Number of Evidence Items",
            height=300
        )
        
        return fig
    
    def _create_confidence_progression(self, voting_results: Dict) -> go.Figure:
        """Create a line chart showing confidence progression through the pipeline."""
        voting_details = voting_results.get("voting_details", {})
        
        stages = ["Individual Votes", "Weighted Votes", "Final Confidence"]
        openai_progression = [
            voting_details.get("openai", {}).get("vote", 0.0),
            voting_details.get("openai", {}).get("vote", 0.0) * voting_details.get("openai", {}).get("weight", 0.0),
            voting_results.get("final_confidence", 0.0)
        ]
        
        gemini_progression = [
            voting_details.get("gemini", {}).get("vote", 0.0),
            voting_details.get("gemini", {}).get("vote", 0.0) * voting_details.get("gemini", {}).get("weight", 0.0),
            voting_results.get("final_confidence", 0.0)
        ]
        
        wikipedia_progression = [
            voting_details.get("wikipedia", {}).get("vote", 0.0),
            voting_details.get("wikipedia", {}).get("vote", 0.0) * voting_details.get("wikipedia", {}).get("weight", 0.0),
            voting_results.get("final_confidence", 0.0)
        ]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=stages,
            y=openai_progression,
            mode='lines+markers',
            name='OpenAI',
            line=dict(color='blue', width=3),
            marker=dict(size=8)
        ))
        
        fig.add_trace(go.Scatter(
            x=stages,
            y=gemini_progression,
            mode='lines+markers',
            name='Gemini',
            line=dict(color='green', width=3),
            marker=dict(size=8)
        ))
        
        fig.add_trace(go.Scatter(
            x=stages,
            y=wikipedia_progression,
            mode='lines+markers',
            name='Wikipedia',
            line=dict(color='orange', width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title="Confidence Progression Through Pipeline",
            xaxis_title="Pipeline Stages",
            yaxis_title="Confidence Score",
            height=400
        )
        
        return fig
    
    def _create_source_credibility_matrix(self, evidence_set: List[Dict]) -> go.Figure:
        """Create a matrix showing source credibility scores."""
        if not evidence_set:
            return go.Figure()
        
        # Calculate credibility metrics for each source
        source_metrics = {}
        for evidence in evidence_set:
            source = evidence.get("source", "unknown")
            if source not in source_metrics:
                source_metrics[source] = {
                    "count": 0,
                    "total_relevance": 0.0,
                    "total_support": 0.0
                }
            
            source_metrics[source]["count"] += 1
            source_metrics[source]["total_relevance"] += evidence.get("relevance_score", 0.0)
            source_metrics[source]["total_support"] += evidence.get("support_score", 0.0)
        
        # Calculate averages
        sources = []
        avg_relevance = []
        avg_support = []
        counts = []
        
        for source, metrics in source_metrics.items():
            sources.append(source)
            avg_relevance.append(metrics["total_relevance"] / metrics["count"])
            avg_support.append(metrics["total_support"] / metrics["count"])
            counts.append(metrics["count"])
        
        # Create subplot
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Average Relevance by Source", "Average Support by Source"),
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )
        
        fig.add_trace(
            go.Bar(x=sources, y=avg_relevance, name="Relevance", marker_color='lightblue'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=sources, y=avg_support, name="Support", marker_color='lightgreen'),
            row=1, col=2
        )
        
        fig.update_layout(
            title="Source Credibility Matrix",
            height=400,
            showlegend=False
        )
        
        return fig 

    def display_visualizations_in_streamlit(self, visualizations: Dict[str, Any]):
        """Display essential visualizations in Streamlit with clean layout."""
        
        st.markdown("## 📊 Visual Analysis Dashboard")
        
        # Create 2x3 grid layout for 5 visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎯 Confidence & Risk")
            st.plotly_chart(visualizations["confidence_gauge"], use_container_width=True)
            st.plotly_chart(visualizations["risk_thermometer"], use_container_width=True)
        
        with col2:
            st.markdown("### 📈 Source Analysis")
            st.plotly_chart(visualizations["source_contribution_chart"], use_container_width=True)
            st.plotly_chart(visualizations["detection_signals_radar"], use_container_width=True)
        
        # Full width for flowchart
        st.markdown("### 🔄 Decision Process")
        st.plotly_chart(visualizations["consensus_flowchart"], use_container_width=True)
    
    def create_interactive_dashboard(self, explanation: Dict[str, Any]) -> str:
        """Create an interactive HTML dashboard with all visualizations."""
        
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>XAI Hallucination Detection Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .dashboard { max-width: 1200px; margin: 0 auto; }
                .section { margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 10px; }
                .chart-container { margin: 20px 0; }
                .summary-box { background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 10px 0; }
                .metric { display: inline-block; margin: 10px; padding: 10px; background: white; border-radius: 5px; }
                .high-confidence { border-left: 4px solid #28a745; }
                .medium-confidence { border-left: 4px solid #ffc107; }
                .low-confidence { border-left: 4px solid #dc3545; }
            </style>
        </head>
        <body>
            <div class="dashboard">
                <h1>🎓 XAI Hallucination Detection Dashboard</h1>
                
                <div class="section">
                    <h2>📊 Summary Metrics</h2>
                    <div class="summary-box">
        """
        
        # Add summary metrics
        final_verdict = explanation.get("decision_rationale", {}).get("final_verdict", "UNKNOWN")
        final_confidence = explanation.get("decision_rationale", {}).get("final_confidence", 0.0)
        consensus_level = explanation.get("decision_rationale", {}).get("consensus_level", "UNKNOWN")
        
        confidence_class = "high-confidence" if final_confidence > 0.7 else "medium-confidence" if final_confidence > 0.4 else "low-confidence"
        
        html += f"""
                        <div class="metric {confidence_class}">
                            <strong>Final Verdict:</strong> {final_verdict}
                        </div>
                        <div class="metric {confidence_class}">
                            <strong>Confidence:</strong> {final_confidence:.2f}
                        </div>
                        <div class="metric">
                            <strong>Consensus:</strong> {consensus_level}
                        </div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>🎯 Confidence & Risk Assessment</h2>
                    <div class="chart-container" id="confidence-gauge"></div>
                    <div class="chart-container" id="risk-thermometer"></div>
                </div>
                
                <div class="section">
                    <h2>📈 Source Analysis</h2>
                    <div class="chart-container" id="source-contribution"></div>
                    <div class="chart-container" id="evidence-heatmap"></div>
                </div>
                
                <div class="section">
                    <h2>🔍 Detection Signals</h2>
                    <div class="chart-container" id="detection-radar"></div>
                    <div class="chart-container" id="correction-comparison"></div>
                </div>
                
                <div class="section">
                    <h2>🔄 Process Flow</h2>
                    <div class="chart-container" id="consensus-flowchart"></div>
                    <div class="chart-container" id="confidence-progression"></div>
                </div>
            </div>
            
            <script>
                // Add Plotly charts here
                // This would be populated with the actual chart data
                console.log("Dashboard loaded successfully");
            </script>
        </body>
        </html>
        """
        
        return html
    
    def create_visualization_summary(self, visualizations: Dict[str, Any]) -> str:
        """Create a text summary of key visualization insights."""
        
        summary = """
## 📊 Visualization Insights

### 🎯 Confidence Assessment
"""
        
        # Analyze confidence gauge
        confidence_gauge = visualizations.get("confidence_gauge")
        if confidence_gauge:
            # Extract confidence value from the gauge
            confidence_value = confidence_gauge.data[0].value / 100 if hasattr(confidence_gauge.data[0], 'value') else 0.0
            
            if confidence_value > 0.8:
                summary += "• **High Confidence**: The system is very confident in its assessment\n"
            elif confidence_value > 0.6:
                summary += "• **Moderate Confidence**: The system has reasonable confidence in its assessment\n"
            else:
                summary += "• **Low Confidence**: The system has limited confidence due to insufficient evidence\n"
        
        summary += """
### 📈 Source Contributions
"""
        
        # Analyze source contribution chart
        source_chart = visualizations.get("source_contribution_chart")
        if source_chart:
            summary += "• **Source Analysis**: Shows how each source (OpenAI, Gemini, Wikipedia) contributed to the final decision\n"
            summary += "• **Weight Distribution**: Displays the relative importance of each source\n"
        
        summary += """
### 🔍 Detection Signals
"""
        
        # Analyze detection radar
        radar_chart = visualizations.get("detection_signals_radar")
        if radar_chart:
            summary += "• **Detection Performance**: Radar chart shows the strength of different detection signals\n"
            summary += "• **Signal Balance**: Indicates which detection methods were most effective\n"
        
        summary += """
### 🔄 Process Flow
"""
        
        # Analyze consensus flowchart
        flowchart = visualizations.get("consensus_flowchart")
        if flowchart:
            summary += "• **Decision Process**: Flowchart shows the step-by-step decision-making process\n"
            summary += "• **Pipeline Stages**: Visualizes how evidence flows through the system\n"
        
        return summary 