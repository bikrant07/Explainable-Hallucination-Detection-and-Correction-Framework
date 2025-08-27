"""
test_xai_visualizations.py
Test script to demonstrate XAI visualization features with sample data.
"""
import streamlit as st
from xai_explanation import XAIExplanation
import plotly.graph_objects as go

def create_sample_data():
    """Create sample data for testing visualizations."""
    
    # Sample detection results
    detection_results = {
        "signals": {
            "consistency_score": 0.85,
            "stability_score": 0.78,
            "uncertainty_score": 0.23
        },
        "risk_score": 0.15,
        "flagged_spans": [
            ("This claim is accurate", 0.1),
            ("based on educational research", 0.05)
        ]
    }
    
    # Sample correction results
    correction_results = {
        "openai": {
            "verdict": "YES",
            "confidence": 0.85,
            "corrected_claim": "N/A",
            "explanation": "This claim is accurate and well-supported by educational literature."
        },
        "gemini": {
            "verdict": "YES",
            "confidence": 0.78,
            "corrected_claim": "N/A",
            "explanation": "The statement correctly describes constructivism."
        },
        "wikipedia": {
            "verdict": "YES",
            "confidence": 0.72,
            "corrected_claim": "N/A",
            "explanation": "Wikipedia evidence supports this claim."
        }
    }
    
    # Sample evidence set
    evidence_set = [
        {
            "source": "knowledge_base",
            "title": "Constructivism Learning Theory",
            "content": "Constructivism posits that learners construct knowledge through experience and reflection.",
            "relevance_score": 0.95,
            "support_score": 0.92,
            "url": "kb://constructivism"
        },
        {
            "source": "wikipedia",
            "title": "Constructivism (philosophy of education)",
            "content": "Constructivism is a learning theory that suggests students construct knowledge...",
            "relevance_score": 0.88,
            "support_score": 0.85,
            "url": "https://en.wikipedia.org/wiki/Constructivism_(philosophy_of_education)"
        },
        {
            "source": "google",
            "title": "Constructivism in Education",
            "content": "Constructivism in education is a learning theory...",
            "relevance_score": 0.82,
            "support_score": 0.78,
            "url": "https://example.com/constructivism-education"
        }
    ]
    
    # Sample voting results
    voting_results = {
        "final_verdict": "YES",
        "final_confidence": 0.82,
        "consensus_level": "HIGH_CONSENSUS",
        "weights": {
            "openai": 0.4,
            "gemini": 0.4,
            "wikipedia": 0.2
        },
        "voting_details": {
            "openai": {
                "vote": 1.0,
                "confidence": 0.85,
                "weight": 0.4,
                "contribution": 0.34
            },
            "gemini": {
                "vote": 1.0,
                "confidence": 0.78,
                "weight": 0.4,
                "contribution": 0.31
            },
            "wikipedia": {
                "vote": 1.0,
                "confidence": 0.72,
                "weight": 0.2,
                "contribution": 0.14
            }
        }
    }
    
    return detection_results, correction_results, evidence_set, voting_results

def main():
    """Main function to test XAI visualizations."""
    st.set_page_config(page_title="XAI Visualization Test", layout="wide")
    
    st.title("🎓 XAI Visualization Test Dashboard")
    st.markdown("Testing the comprehensive visualization features of the XAI explanation module.")
    
    # Create sample data
    detection_results, correction_results, evidence_set, voting_results = create_sample_data()
    
    # Initialize XAI explanation module
    xai = XAIExplanation()
    
    # Generate complete explanation with visualizations
    explanation = xai.generate_complete_explanation(
        original_output="Constructivism is a learning theory that suggests students construct knowledge through experience and reflection.",
        query="What is constructivism in education?",
        detection_results=detection_results,
        correction_results=correction_results,
        evidence_set=evidence_set,
        voting_results=voting_results
    )
    
    # Display summary
    st.markdown("## 📊 Analysis Summary")
    st.markdown(explanation["summary"])
    
    # Display visualizations
    st.markdown("## 📈 Visualizations")
    xai.display_visualizations_in_streamlit(explanation["visualizations"])
    
    # Display step-by-step explanation
    st.markdown("## 🔍 Step-by-Step Analysis")
    for step in explanation["step_by_step"]:
        with st.expander(f"{step['icon']} Step {step['step']}: {step['title']}"):
            st.markdown(f"**Description:** {step['description']}")
            st.markdown("**Details:**")
            for key, value in step['details'].items():
                if isinstance(value, float):
                    st.markdown(f"- **{key.replace('_', ' ').title()}:** {value:.2f}")
                else:
                    st.markdown(f"- **{key.replace('_', ' ').title()}:** {value}")
    
    # Display evidence attribution
    st.markdown("## 📚 Evidence Attribution")
    attribution = explanation["evidence_attribution"]
    st.markdown(f"**Total Sources:** {attribution['total_sources']}")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Source Breakdown:**")
        for source, count in attribution['source_breakdown'].items():
            st.markdown(f"- {source.title()}: {count} sources")
    
    with col2:
        st.markdown("**Top Supporting Evidence:**")
        for evidence in attribution['top_supporting'][:3]:
            st.markdown(f"- {evidence['title']} (relevance: {evidence['relevance_score']:.2f})")
    
    # Display correction explanation
    st.markdown("## ✏️ Correction Analysis")
    correction_explanation = explanation["correction_explanation"]
    if correction_explanation['has_corrections']:
        for correction in correction_explanation['corrections']:
            st.markdown(f"**{correction['source']} Correction:**")
            st.markdown(f"- **Original:** {correction['original']}")
            st.markdown(f"- **Corrected:** {correction['corrected']}")
            st.markdown(f"- **Rationale:** {correction['rationale']}")
    else:
        st.markdown("✅ No corrections were needed. All verification sources confirmed the accuracy.")
    
    # Display risk assessment
    st.markdown("## ⚠️ Risk Assessment")
    risk = explanation["risk_assessment"]
    st.markdown(f"**Risk Level:** {risk['risk_level']} ({risk['overall_risk']:.2f})")
    
    st.markdown("**Risk Factors:**")
    for factor in risk['risk_factors']:
        st.markdown(f"- {factor}")
    
    if risk['mitigation_suggestions']:
        st.markdown("**Mitigation Suggestions:**")
        for suggestion in risk['mitigation_suggestions']:
            st.markdown(f"- {suggestion}")
    
    # Display recommendations
    st.markdown("## 💡 Recommendations")
    for recommendation in explanation["recommendations"]:
        st.markdown(f"- {recommendation}")
    
    # Display visualization insights
    st.markdown("## 📊 Visualization Insights")
    viz_summary = xai.create_visualization_summary(explanation["visualizations"])
    st.markdown(viz_summary)
    
    # Display what visualizations are available
    st.markdown("## 🎨 Available Visualizations")
    st.markdown("""
    The XAI module now provides **5 essential visualizations**:
    
    1. **🎯 Confidence Gauge** - Shows final confidence score with color-coded zones
    2. **📊 Source Contribution Chart** - Displays how each source contributed to the decision
    3. **🎯 Detection Signals Radar** - Shows the strength of different detection methods
    4. **🌡️ Risk Thermometer** - Visual risk assessment with color coding
    5. **🔄 Consensus Flowchart** - Step-by-step decision process visualization
    
    These visualizations provide a complete understanding of the hallucination detection and correction process.
    """)

if __name__ == "__main__":
    main() 