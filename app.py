"""
app.py
Streamlit UI for advanced hallucination detection system with hybrid retrieval and OpenAI verification.
"""
import streamlit as st
from detect import (
    self_consistency_check, uncertainty_score, load_paraphraser, metamorphic_check, openai_fact_check, openai_available,
    generate_multiple_openai_completions, consensus_check
)
from retrieval import EducationFactRetriever, cross_check
from flag import aggregate_risk, highlight_conflicts, explain_risk
from verify import verify_claim_with_openai, verify_claim_with_gemini, format_verification_result
from wikipedia_evidence import verify_claim_with_wikipedia, format_wikipedia_result
from voting_system import combine_evidence, format_voting_result, get_voting_summary
import nltk
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
nltk.download('punkt', quiet=True)
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="Education Hallucination Detector", layout="wide")
st.title("🎓 Education-Specific LLM Hallucination Detection")

st.markdown("""
**A Context-Aware Explainable Framework for Detecting, Correcting, and Explaining Hallucinations in Educational Content**
""")

# Sidebar: Model selection and API key
def safe_get_api_key():
    key = st.sidebar.text_input("OpenAI API Key", type="password")
    if key:
        return key
    return OPENAI_API_KEY

st.sidebar.header("Education Detection Settings")
# Remove OpenAI usage checkbox and always use OpenAI and Gemini for fact-checking
openai_model = st.sidebar.selectbox("OpenAI Model", ["gpt-3.5-turbo"], index=0)
use_education_knowledge = st.sidebar.checkbox("Use Education Knowledge Base", value=True)
use_wikipedia = st.sidebar.checkbox("Use Wikipedia Evidence Collection", value=True)
# Remove use_verification and always use both LLMs
api_key = OPENAI_API_KEY

st.sidebar.header("Voting Weights")
openai_weight = st.sidebar.slider("OpenAI Weight", 0.0, 1.0, 0.4, 0.1)
gemini_weight = st.sidebar.slider("Gemini Weight", 0.0, 1.0, 0.4, 0.1)
wikipedia_weight = st.sidebar.slider("Wikipedia Weight", 0.0, 1.0, 0.2, 0.1)

# Normalize weights
total_weight = openai_weight + gemini_weight + wikipedia_weight
if total_weight > 0:
    openai_weight /= total_weight
    gemini_weight /= total_weight
    wikipedia_weight /= total_weight

voting_weights = {
    "openai": openai_weight,
    "gemini": gemini_weight,
    "wikipedia": wikipedia_weight
}

st.sidebar.header("Advanced")
min_similarity = st.sidebar.slider("Minimum Fact Similarity", 0.1, 0.9, 0.3, 0.1)
# Remove use_verification and show_openai_factcheck logic in main
consensus_threshold = st.sidebar.slider("Consensus Similarity Threshold", 0.3, 0.9, 0.5, 0.05)

query = st.text_area("Enter the original query (prompt given to the LLM):", "What is constructivism in education?")
llm_output = st.text_area("Paste the LLM output to analyze:", "Constructivism is a learning theory that suggests students construct knowledge through experience and reflection rather than receiving it passively.")
run = st.button("Analyze")

if run and query.strip() and llm_output.strip():
    with st.spinner("Analyzing output with consensus and LLM fact-checking..."):
        para_model, para_tokenizer = (None, None)
        if not api_key: # Only load paraphraser if API key is not available
            para_model, para_tokenizer = load_paraphraser()
        facts = []
        flagged_local = []
        flagged_openai = []
        completions, consensus = [], None
        # Always use OpenAI completions and fact-checking
        if api_key:
            try:
                completions = generate_multiple_openai_completions(query, n=3, api_key=api_key, openai_model=openai_model)
            except Exception as e:
                completions = [f"[OpenAI Completion Error: {e}]"]
        # Remove DuckDuckGo search
        search_snippets = []
        if completions:
            consensus = consensus_check(llm_output, completions, [], threshold=consensus_threshold)
        # Collect evidence from all three sources
        verification_result = None
        try:
            verification_result = verify_claim_with_openai(llm_output, completions, api_key, openai_model)
        except Exception as e:
            verification_result = {
                "verdict": "ERROR",
                "explanation": f"Verification error: {str(e)}",
                "corrected_claim": "N/A",
                "confidence": 0.0
            }
        
        gemini_result = verify_claim_with_gemini(llm_output, completions, GEMINI_API_KEY)
        if not gemini_result.get("confidence"):
            gemini_result["confidence"] = 0.5  # Default confidence
        
        # Wikipedia evidence collection
        wikipedia_result = None
        if use_wikipedia:
            try:
                wikipedia_result = verify_claim_with_wikipedia(llm_output, query)
            except Exception as e:
                wikipedia_result = {
                    "verdict": "ERROR",
                    "explanation": f"Wikipedia error: {str(e)}",
                    "confidence": 0.0
                }
        else:
            wikipedia_result = {
                "verdict": "UNKNOWN",
                "explanation": "Wikipedia evidence collection disabled",
                "confidence": 0.0
            }
        meta_score = None
        try:
            meta_score = metamorphic_check(
                query, None, None, para_model, para_tokenizer,
                use_openai=True, openai_model=openai_model, api_key=api_key
            )
        except Exception:
            meta_score = None
        from nltk.tokenize import sent_tokenize
        n_total = len(sent_tokenize(llm_output))
        n_flagged = len(flagged_local) + len(flagged_openai)
        # Perform voting to combine evidence from all sources
        voting_result = combine_evidence(
            verification_result, 
            gemini_result, 
            wikipedia_result,
            weights=voting_weights
        )
        
        # Extract voting results
        final_verdict = voting_result["final_verdict"]
        final_confidence = voting_result["final_confidence"]
        consensus_level = voting_result["consensus_level"]
        
        # Get individual verdicts for backward compatibility
        openai_verdict = verification_result["verdict"] if verification_result and "verdict" in verification_result else None
        gemini_verdict = gemini_result["verdict"] if gemini_result and "verdict" in gemini_result else None
        wikipedia_verdict = wikipedia_result["verdict"] if wikipedia_result and "verdict" in wikipedia_result else None
        
        # Extract corrections
        corrected = None
        correction_rationale = None
        gemini_correction = None
        gemini_rationale = None
        if verification_result and verification_result.get("corrected_claim") and verification_result["corrected_claim"] != "N/A":
            corrected = verification_result["corrected_claim"]
            correction_rationale = verification_result.get("explanation", "")
        if gemini_result and gemini_result.get("corrected_claim") and gemini_result["corrected_claim"] != "N/A":
            gemini_correction = gemini_result["corrected_claim"]
            gemini_rationale = gemini_result.get("explanation", "")
        
        # Calculate risk based on voting result
        risk = aggregate_risk(
            1.0, 0.5, meta_score if meta_score is not None else 1.0, n_flagged, n_total,
            consensus=consensus,
            openai_verdict=final_verdict,  # Use final verdict from voting
            gemini_verdict=final_verdict,  # Use final verdict from voting
            openai_correction=corrected,
            gemini_correction=gemini_correction,
            user_output=llm_output
        )
        risk_exp = explain_risk(risk)
        highlighted = highlight_conflicts(llm_output, flagged_local)
        for sent in flagged_openai:
            if sent.strip() and sent.strip() in highlighted:
                highlighted = highlighted.replace(sent.strip(), f'<span style="background-color:#ff9999">{sent.strip()}</span>')
        corrected = None
        correction_rationale = None
        gemini_correction = None
        gemini_rationale = None
        if verification_result and verification_result.get("corrected_claim") and verification_result["corrected_claim"] != "N/A":
            corrected = verification_result["corrected_claim"]
            correction_rationale = verification_result.get("explanation", "")
        if gemini_result and gemini_result.get("corrected_claim") and gemini_result["corrected_claim"] != "N/A":
            gemini_correction = gemini_result["corrected_claim"]
            gemini_rationale = gemini_result.get("explanation", "")
        # Display voting result
        st.markdown(format_voting_result(voting_result), unsafe_allow_html=True)
        
        # Display risk assessment
        verdict = "Hallucination Detected" if risk > 0.5 else "No Hallucination Detected"
        if risk > 0.5:
            bg_color = "#ffe8e8"  # Light red background
            text_color = "#5a2d2d"  # Dark red text
        else:
            bg_color = "#e8f5e8"  # Light green background
            text_color = "#2d5a2d"  # Dark green text
        
        st.markdown(f"""
        <div style='padding:1em;background-color:{bg_color};border-radius:10px;margin-bottom:1em;border-left:4px solid {text_color};'>
        <h3 style='margin-bottom:0.5em;color:{text_color};'>Risk Assessment: {verdict}</h3>
        <b style='color:{text_color};'>Risk Score:</b> <span style='color:{text_color};'>{risk:.2f}</span><br>
        <b style='color:{text_color};'>Final Confidence:</b> <span style='color:{text_color};'>{final_confidence:.2f}</span><br>
        <b style='color:{text_color};'>Consensus Level:</b> <span style='color:{text_color};'>{consensus_level}</span><br>
        <b style='color:{text_color};'>Correction (OpenAI):</b> <span style='color:{text_color};'>{corrected if corrected else 'N/A'}</span><br>
        <b style='color:{text_color};'>Correction (Gemini):</b> <span style='color:{text_color};'>{gemini_correction if gemini_correction else 'N/A'}</span><br>
        <b style='color:{text_color};'>Explanation:</b> <span style='color:{text_color};'>{risk_exp}</span>
        </div>
        """, unsafe_allow_html=True)
        tabs = st.tabs(["Voting Results", "Detection", "Correction", "Explanation", "Evidence"])
        with tabs[0]:
            st.subheader("Voting Results")
            
            # Display voting summary
            voting_summary = get_voting_summary(voting_result)
            st.markdown("**Final Verdict:** " + voting_summary["final_verdict"])
            st.markdown("**Final Confidence:** " + f"{voting_summary['final_confidence']:.2f}")
            st.markdown("**Consensus Level:** " + voting_summary["consensus_level"])
            
            # Display individual source results
            st.subheader("Individual Source Results")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**OpenAI**")
                st.markdown(f"Verdict: {openai_verdict}")
                st.markdown(f"Confidence: {verification_result.get('confidence', 0.0):.2f}")
                st.markdown(f"Weight: {voting_weights['openai']:.2f}")
            
            with col2:
                st.markdown("**Gemini**")
                st.markdown(f"Verdict: {gemini_verdict}")
                st.markdown(f"Confidence: {gemini_result.get('confidence', 0.0):.2f}")
                st.markdown(f"Weight: {voting_weights['gemini']:.2f}")
            
            with col3:
                st.markdown("**Wikipedia**")
                st.markdown(f"Verdict: {wikipedia_verdict}")
                st.markdown(f"Confidence: {wikipedia_result.get('confidence', 0.0):.2f}")
                st.markdown(f"Weight: {voting_weights['wikipedia']:.2f}")
            

        
        with tabs[1]:
            st.subheader("LLM Output (Analyzed)")
            st.markdown(highlighted, unsafe_allow_html=True)
            st.caption(f"Metamorphic stability: {meta_score if meta_score is not None else 'N/A'}")
            if completions:
                st.subheader("Consensus Check (LLM)")
                st.markdown("**OpenAI LLM Completions:**")
                for i, comp in enumerate(completions):
                    st.code(comp, language="text")
                if consensus:
                    st.markdown(f"**LLM Agreement:** {'✅' if consensus['llm_agree'] else '❌'}")
                # XAI: Similarity bar chart
                if consensus and hasattr(consensus, 'sims'):
                    scores = consensus['sims']
                else:
                    # Compute similarity scores for bar chart
                    from sentence_transformers import SentenceTransformer
                    from sklearn.metrics.pairwise import cosine_similarity
                    embedder = SentenceTransformer('all-MiniLM-L6-v2')
                    user_emb = embedder.encode([llm_output])[0]
                    comp_embs = embedder.encode(completions)
                    scores = cosine_similarity([user_emb], comp_embs)[0]
                labels = [f"LLM {i+1}" for i in range(len(completions))]
                fig, ax = plt.subplots()
                ax.bar(labels, scores, color='skyblue')
                ax.set_ylabel('Similarity')
                ax.set_ylim(0, 1)
                ax.set_title('Similarity to User Output')
                st.pyplot(fig)
        with tabs[1]:
            st.subheader("Correction Suggestions")
            if corrected:
                import difflib
                diff = difflib.ndiff(llm_output.split(), corrected.split())
                html = ' '.join([
                    f"<span style='background-color:#ccffcc'>{w[2:]}</span>" if w.startswith('+ ') else
                    f"<span style='background-color:#ffcccc'>{w[2:]}</span>" if w.startswith('- ') else
                    w[2:] for w in diff if not w.startswith('?')
                ])
                st.markdown(f"**OpenAI Correction:**<br>{html}", unsafe_allow_html=True)
                if correction_rationale:
                    st.info(f"OpenAI Rationale: {correction_rationale}")
            else:
                st.info("No correction needed or available from OpenAI.")
            if gemini_correction:
                import difflib
                diff = difflib.ndiff(llm_output.split(), gemini_correction.split())
                html = ' '.join([
                    f"<span style='background-color:#ccffcc'>{w[2:]}</span>" if w.startswith('+ ') else
                    f"<span style='background-color:#ffcccc'>{w[2:]}</span>" if w.startswith('- ') else
                    w[2:] for w in diff if not w.startswith('?')
                ])
                st.markdown(f"**Gemini Correction:**<br>{html}", unsafe_allow_html=True)
                if gemini_rationale:
                    st.info(f"Gemini Rationale: {gemini_rationale}")
            else:
                st.info("No correction needed or available from Gemini.")
        with tabs[2]:
            st.subheader("Correction")
            if corrected:
                st.markdown("**OpenAI Correction:**")
                st.markdown(f"**Original:** {llm_output}")
                st.markdown(f"**Corrected:** {corrected}")
                st.markdown(f"**Rationale:** {correction_rationale}")
            if gemini_correction:
                st.markdown("**Gemini Correction:**")
                st.markdown(f"**Original:** {llm_output}")
                st.markdown(f"**Corrected:** {gemini_correction}")
                st.markdown(f"**Rationale:** {gemini_rationale}")
            if not corrected and not gemini_correction:
                st.markdown("**No corrections suggested by the models.**")
        
        with tabs[3]:
            st.subheader("Explanation")
            st.markdown(f"**Why was this flagged?**<br>{risk_exp}", unsafe_allow_html=True)
            if verification_result:
                st.markdown(format_verification_result(verification_result, provider="OpenAI"))
            if gemini_result:
                st.markdown(format_verification_result(gemini_result, provider="Gemini"))
            if wikipedia_result:
                st.markdown(format_wikipedia_result(wikipedia_result, provider="Wikipedia"))
            if consensus:
                st.markdown("**Consensus Reasoning:**")
                st.markdown(f"LLM Agreement: {'Yes' if consensus['llm_agree'] else 'No'}")
            # XAI: Decision flowchart
            st.subheader("Decision Flowchart")
            st.graphviz_chart('''
digraph G {
    UserOutput [label="User Output"]
    OpenAI [label="OpenAI Verdict"]
    Gemini [label="Gemini Verdict"]
    Wikipedia [label="Wikipedia Evidence"]
    Voting [label="Voting System"]
    Consensus [label="LLM Consensus"]
    Risk [label="Final Risk Score"]
    UserOutput -> OpenAI
    UserOutput -> Gemini
    UserOutput -> Wikipedia
    OpenAI -> Voting
    Gemini -> Voting
    Wikipedia -> Voting
    Voting -> Risk
    Consensus -> Risk
}
''')
        
        with tabs[4]:
            st.subheader("Evidence Used")
            if completions:
                st.markdown("**LLM Completions:**")
                for i, comp in enumerate(completions):
                    st.markdown(f"- {comp}")
            
            # Display Wikipedia evidence
            if wikipedia_result and wikipedia_result.get("sources"):
                st.markdown("**Wikipedia Sources:**")
                for source in wikipedia_result["sources"][:5]:  # Show top 5 sources
                    st.markdown(f"- [{source['title']}]({source['url']}) (relevance: {source['relevance']:.2f}, support: {source['support']:.2f})")
            
            # Display voting details
            st.markdown("**Voting Details:**")
            voting_details = voting_result.get("voting_details", {})
            for source, details in voting_details.items():
                st.markdown(f"- **{source.upper()}**: Vote={details['vote']:.2f}, Confidence={details['confidence']:.2f}, Weight={details['weight']:.2f}, Contribution={details['contribution']:.2f}")
else:
    st.info("Enter a query and LLM output, then click Analyze to begin.")
