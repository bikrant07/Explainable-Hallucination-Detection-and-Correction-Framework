"""
test_voting_system.py
Test script for Wikipedia evidence collection and voting system.
"""
import os
from dotenv import load_dotenv
from wikipedia_evidence import verify_claim_with_wikipedia, format_wikipedia_result
from voting_system import combine_evidence, format_voting_result, get_voting_summary
from verify import verify_claim_with_openai, verify_claim_with_gemini

load_dotenv()

def test_wikipedia_evidence():
    """Test Wikipedia evidence collection."""
    print("🌐 Testing Wikipedia Evidence Collection")
    print("=" * 50)
    
    test_cases = [
        {
            "query": "constructivism learning theory",
            "claim": "Constructivism is a learning theory that suggests students construct knowledge through experience and reflection.",
            "expected": "YES"
        },
        {
            "query": "Common Core State Standards",
            "claim": "Common Core standards are weekly tests that students must pass with 100% accuracy.",
            "expected": "NO"
        },
        {
            "query": "STEM education",
            "claim": "STEM education integrates Science, Technology, Engineering, and Mathematics in an interdisciplinary approach.",
            "expected": "YES"
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n📚 Test Case {i}: {case['query']}")
        print("-" * 30)
        
        try:
            result = verify_claim_with_wikipedia(case['claim'], case['query'])
            
            print(f"Verdict: {result['verdict']}")
            print(f"Confidence: {result['confidence']:.2f}")
            print(f"Explanation: {result['explanation']}")
            
            if result.get('sources'):
                print(f"Sources found: {len(result['sources'])}")
                for source in result['sources'][:2]:  # Show top 2 sources
                    print(f"  - {source['title']} (relevance: {source['relevance']:.2f})")
            
            # Check if verdict matches expectation
            if result['verdict'] == case['expected']:
                print("✅ PASSED - Verdict matches expectation")
            else:
                print("⚠️ PARTIAL - Verdict differs from expectation")
                
        except Exception as e:
            print(f"❌ ERROR: {e}")

def test_voting_system():
    """Test the voting system."""
    print("\n🗳️ Testing Voting System")
    print("=" * 50)
    
    # Test case 1: All sources agree
    print("\n📊 Test Case 1: All Sources Agree (YES)")
    print("-" * 40)
    
    openai_result = {
        "verdict": "YES",
        "confidence": 0.8,
        "explanation": "OpenAI supports this claim."
    }
    
    gemini_result = {
        "verdict": "YES",
        "confidence": 0.7,
        "explanation": "Gemini also supports this claim."
    }
    
    wikipedia_result = {
        "verdict": "YES",
        "confidence": 0.6,
        "explanation": "Wikipedia evidence supports this claim."
    }
    
    voting_result = combine_evidence(openai_result, gemini_result, wikipedia_result)
    
    print(f"Final Verdict: {voting_result['final_verdict']}")
    print(f"Final Confidence: {voting_result['final_confidence']:.2f}")
    print(f"Consensus Level: {voting_result['consensus_level']}")
    print(f"Explanation: {voting_result['explanation']}")
    
    # Test case 2: Sources disagree
    print("\n📊 Test Case 2: Sources Disagree")
    print("-" * 40)
    
    openai_result = {
        "verdict": "YES",
        "confidence": 0.8,
        "explanation": "OpenAI supports this claim."
    }
    
    gemini_result = {
        "verdict": "NO",
        "confidence": 0.7,
        "explanation": "Gemini does not support this claim."
    }
    
    wikipedia_result = {
        "verdict": "UNKNOWN",
        "confidence": 0.5,
        "explanation": "Wikipedia has limited information."
    }
    
    voting_result = combine_evidence(openai_result, gemini_result, wikipedia_result)
    
    print(f"Final Verdict: {voting_result['final_verdict']}")
    print(f"Final Confidence: {voting_result['final_confidence']:.2f}")
    print(f"Consensus Level: {voting_result['consensus_level']}")
    print(f"Explanation: {voting_result['explanation']}")

def test_integrated_system():
    """Test the integrated system with real API calls."""
    print("\n🔗 Testing Integrated System")
    print("=" * 50)
    
    # Check API keys
    openai_api_key = os.getenv("OPENAI_API_KEY")
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    
    if not openai_api_key or not gemini_api_key:
        print("⚠️ API keys not found. Skipping integrated test.")
        return
    
    test_case = {
        "query": "constructivism learning theory",
        "claim": "Constructivism is a learning theory that suggests students construct knowledge through experience and reflection."
    }
    
    print(f"Testing: {test_case['claim']}")
    print("-" * 40)
    
    try:
        # Get OpenAI result
        print("🤖 Getting OpenAI result...")
        openai_result = verify_claim_with_openai(test_case['claim'], [], openai_api_key)
        print(f"OpenAI: {openai_result['verdict']} (confidence: {openai_result.get('confidence', 0.0):.2f})")
        
        # Get Gemini result
        print("🤖 Getting Gemini result...")
        gemini_result = verify_claim_with_gemini(test_case['claim'], [], gemini_api_key)
        print(f"Gemini: {gemini_result['verdict']} (confidence: {gemini_result.get('confidence', 0.0):.2f})")
        
        # Get Wikipedia result
        print("🌐 Getting Wikipedia result...")
        wikipedia_result = verify_claim_with_wikipedia(test_case['claim'], test_case['query'])
        print(f"Wikipedia: {wikipedia_result['verdict']} (confidence: {wikipedia_result['confidence']:.2f})")
        
        # Combine evidence
        print("🗳️ Combining evidence...")
        voting_result = combine_evidence(openai_result, gemini_result, wikipedia_result)
        
        print(f"\n🎯 FINAL RESULT:")
        print(f"Verdict: {voting_result['final_verdict']}")
        print(f"Confidence: {voting_result['final_confidence']:.2f}")
        print(f"Consensus: {voting_result['consensus_level']}")
        
        # Display voting summary
        summary = get_voting_summary(voting_result)
        print(f"\n📊 Voting Summary:")
        for source, details in summary['source_summary'].items():
            print(f"  {source.upper()}: {details['vote']} (confidence: {details['confidence']:.2f}, weight: {details['weight']:.2f})")
        
    except Exception as e:
        print(f"❌ Error in integrated test: {e}")

def main():
    """Run all tests."""
    print("🧪 Testing Wikipedia Evidence Collection and Voting System")
    print("=" * 70)
    
    # Test Wikipedia evidence collection
    test_wikipedia_evidence()
    
    # Test voting system
    test_voting_system()
    
    # Test integrated system
    test_integrated_system()
    
    print("\n🎉 All tests completed!")

if __name__ == "__main__":
    main() 