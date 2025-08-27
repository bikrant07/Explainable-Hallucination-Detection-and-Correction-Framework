"""
wikipedia_evidence.py
Implements Wikipedia evidence collection for educational content verification.
Provides structured evidence gathering from Wikipedia articles and educational resources.
"""
import os
import re
import requests
from typing import List, Dict, Any, Optional
from urllib.parse import quote
import json

class WikipediaEvidenceCollector:
    """
    Collects evidence from Wikipedia for educational content verification.
    """
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Education-Hallucination-Detector/1.0 (Educational Research)'
        })
        
    def search_wikipedia(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search Wikipedia for educational content related to the query.
        """
        try:
            # Wikipedia API search endpoint
            search_url = "https://en.wikipedia.org/w/api.php"
            params = {
                'action': 'query',
                'format': 'json',
                'list': 'search',
                'srsearch': query,
                'srlimit': max_results,
                'srnamespace': 0,  # Main namespace only
                'srprop': 'snippet|title|timestamp'
            }
            
            response = self.session.get(search_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            if 'query' in data and 'search' in data['query']:
                return data['query']['search']
            return []
            
        except Exception as e:
            print(f"Wikipedia search error: {e}")
            return []
    
    def get_wikipedia_content(self, page_id: str) -> Optional[Dict[str, Any]]:
        """
        Get full content of a Wikipedia page.
        """
        try:
            # Wikipedia API content endpoint
            content_url = "https://en.wikipedia.org/w/api.php"
            params = {
                'action': 'query',
                'format': 'json',
                'prop': 'extracts|info',
                'pageids': page_id,
                'exintro': True,  # Only introduction
                'explaintext': True,  # Plain text
                'inprop': 'url'
            }
            
            response = self.session.get(content_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            if 'query' in data and 'pages' in data['query']:
                pages = data['query']['pages']
                if page_id in pages:
                    return pages[page_id]
            return None
            
        except Exception as e:
            print(f"Wikipedia content error: {e}")
            return None
    
    def extract_educational_evidence(self, query: str, claim: str) -> Dict[str, Any]:
        """
        Extract educational evidence from Wikipedia for a given claim.
        """
        evidence = {
            "wikipedia_articles": [],
            "supporting_evidence": [],
            "contradicting_evidence": [],
            "neutral_evidence": [],
            "verdict": "UNKNOWN",
            "confidence": 0.0,
            "sources": []
        }
        
        try:
            # Search for educational content
            search_results = self.search_wikipedia(query, max_results=5)
            
            for result in search_results:
                page_id = result['pageid']
                title = result['title']
                snippet = result['snippet']
                
                # Get full content
                content = self.get_wikipedia_content(str(page_id))
                if content:
                    extract = content.get('extract', '')
                    url = content.get('fullurl', '')
                    
                    article_evidence = {
                        "title": title,
                        "url": url,
                        "snippet": snippet,
                        "extract": extract,
                        "relevance_score": self._calculate_relevance(query, title, snippet),
                        "support_score": self._calculate_support(claim, extract)
                    }
                    
                    evidence["wikipedia_articles"].append(article_evidence)
                    evidence["sources"].append({
                        "type": "wikipedia",
                        "title": title,
                        "url": url,
                        "relevance": article_evidence["relevance_score"],
                        "support": article_evidence["support_score"]
                    })
                    
                    # Categorize evidence
                    if article_evidence["support_score"] > 0.7:
                        evidence["supporting_evidence"].append(article_evidence)
                    elif article_evidence["support_score"] < 0.3:
                        evidence["contradicting_evidence"].append(article_evidence)
                    else:
                        evidence["neutral_evidence"].append(article_evidence)
            
            # Determine verdict based on evidence
            evidence["verdict"], evidence["confidence"] = self._determine_verdict(evidence)
            
        except Exception as e:
            print(f"Wikipedia evidence extraction error: {e}")
            evidence["verdict"] = "ERROR"
            evidence["confidence"] = 0.0
        
        return evidence
    
    def _calculate_relevance(self, query: str, title: str, snippet: str) -> float:
        """
        Calculate relevance score between query and Wikipedia content.
        """
        query_terms = set(re.findall(r'\b\w+\b', query.lower()))
        title_terms = set(re.findall(r'\b\w+\b', title.lower()))
        snippet_terms = set(re.findall(r'\b\w+\b', snippet.lower()))
        
        # Calculate overlap
        title_overlap = len(query_terms.intersection(title_terms)) / len(query_terms) if query_terms else 0
        snippet_overlap = len(query_terms.intersection(snippet_terms)) / len(query_terms) if query_terms else 0
        
        # Weighted relevance score
        relevance = (title_overlap * 0.6) + (snippet_overlap * 0.4)
        return min(relevance, 1.0)
    
    def _calculate_support(self, claim: str, content: str) -> float:
        """
        Calculate how well Wikipedia content supports the claim.
        """
        # Simple keyword-based support calculation
        claim_terms = set(re.findall(r'\b\w+\b', claim.lower()))
        content_terms = set(re.findall(r'\b\w+\b', content.lower()))
        
        # Calculate term overlap
        overlap = len(claim_terms.intersection(content_terms))
        total_claim_terms = len(claim_terms)
        
        if total_claim_terms == 0:
            return 0.0
        
        # Normalize support score
        support_score = overlap / total_claim_terms
        return min(support_score, 1.0)
    
    def _determine_verdict(self, evidence: Dict[str, Any]) -> tuple:
        """
        Determine verdict and confidence based on collected evidence.
        """
        supporting_count = len(evidence["supporting_evidence"])
        contradicting_count = len(evidence["contradicting_evidence"])
        neutral_count = len(evidence["neutral_evidence"])
        total_articles = len(evidence["wikipedia_articles"])
        
        if total_articles == 0:
            return "UNKNOWN", 0.0
        
        # Calculate confidence based on evidence quality
        avg_relevance = sum(article["relevance_score"] for article in evidence["wikipedia_articles"]) / total_articles
        avg_support = sum(article["support_score"] for article in evidence["wikipedia_articles"]) / total_articles
        
        # Determine verdict
        if supporting_count > contradicting_count and avg_support > 0.5:
            verdict = "YES"
            confidence = min((supporting_count / total_articles) * avg_support, 1.0)
        elif contradicting_count > supporting_count and avg_support < 0.3:
            verdict = "NO"
            confidence = min((contradicting_count / total_articles) * (1 - avg_support), 1.0)
        else:
            verdict = "UNKNOWN"
            confidence = avg_relevance * 0.5
        
        return verdict, confidence

def verify_claim_with_wikipedia(claim: str, query: str) -> Dict[str, Any]:
    """
    Verify a claim using Wikipedia evidence.
    """
    collector = WikipediaEvidenceCollector()
    evidence = collector.extract_educational_evidence(query, claim)
    
    return {
        "verdict": evidence["verdict"],
        "confidence": evidence["confidence"],
        "explanation": _format_wikipedia_explanation(evidence),
        "sources": evidence["sources"],
        "supporting_evidence": evidence["supporting_evidence"],
        "contradicting_evidence": evidence["contradicting_evidence"]
    }

def _format_wikipedia_explanation(evidence: Dict[str, Any]) -> str:
    """
    Format Wikipedia evidence into an explanation.
    """
    if evidence["verdict"] == "ERROR":
        return "Error collecting Wikipedia evidence."
    
    if not evidence["wikipedia_articles"]:
        return "No relevant Wikipedia articles found."
    
    explanation_parts = []
    
    if evidence["supporting_evidence"]:
        explanation_parts.append(f"Found {len(evidence['supporting_evidence'])} supporting Wikipedia articles.")
    
    if evidence["contradicting_evidence"]:
        explanation_parts.append(f"Found {len(evidence['contradicting_evidence'])} contradicting Wikipedia articles.")
    
    if evidence["neutral_evidence"]:
        explanation_parts.append(f"Found {len(evidence['neutral_evidence'])} neutral Wikipedia articles.")
    
    # Add top supporting evidence
    if evidence["supporting_evidence"]:
        top_support = max(evidence["supporting_evidence"], key=lambda x: x["support_score"])
        explanation_parts.append(f"Top supporting source: {top_support['title']} (support score: {top_support['support_score']:.2f})")
    
    # Add top contradicting evidence
    if evidence["contradicting_evidence"]:
        top_contradict = max(evidence["contradicting_evidence"], key=lambda x: x["support_score"])
        explanation_parts.append(f"Top contradicting source: {top_contradict['title']} (support score: {top_contradict['support_score']:.2f})")
    
    explanation_parts.append(f"Overall confidence: {evidence['confidence']:.2f}")
    
    return " ".join(explanation_parts)

def format_wikipedia_result(result: Dict[str, Any], provider: str = "Wikipedia") -> str:
    """
    Format Wikipedia verification result for display.
    """
    verdict = result.get("verdict", "UNKNOWN")
    confidence = result.get("confidence", 0.0)
    explanation = result.get("explanation", "")
    sources = result.get("sources", [])
    
    if verdict == "YES":
        status = f"✅ SUPPORTED by {provider}"
    elif verdict == "NO":
        status = f"❌ NOT SUPPORTED by {provider}"
    else:
        status = f"❓ UNKNOWN ({provider})"
    
    output = f"""
**{status}**

**Confidence:** {confidence:.2f}

**Explanation:** {explanation}

"""
    
    if sources:
        output += "**Sources:**\n"
        for source in sources[:3]:  # Show top 3 sources
            output += f"- [{source['title']}]({source['url']}) (relevance: {source['relevance']:.2f})\n"
    
    return output

# Example usage
if __name__ == "__main__":
    # Test Wikipedia evidence collection
    query = "constructivism learning theory"
    claim = "Constructivism is a learning theory that suggests students construct knowledge through experience and reflection."
    
    result = verify_claim_with_wikipedia(claim, query)
    print("Wikipedia Verification Result:")
    print(json.dumps(result, indent=2)) 