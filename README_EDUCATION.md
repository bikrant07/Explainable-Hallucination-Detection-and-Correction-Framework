# 🎓 Education-Specific LLM Hallucination Detection System

A specialized framework for detecting, correcting, and explaining hallucinations in educational content, curriculum materials, and pedagogical discussions.

## 🌟 Features

### Education-Specific Focus
- **Domain-Specialized Knowledge Base**: Curated educational facts from multiple sources
- **Curriculum-Aware Detection**: Understands educational standards and frameworks
- **Pedagogical Accuracy**: Validates teaching methods and learning theories
- **Academic Content Verification**: Specialized for educational materials

### Comprehensive Detection Methods
- **Multi-Model Verification**: Uses both OpenAI GPT and Google Gemini for fact-checking
- **Hybrid Retrieval**: Combines dense (FAISS) and sparse (Whoosh) search
- **Self-Consistency Analysis**: Checks output consistency across multiple generations
- **Metamorphic Testing**: Evaluates stability under input variations
- **Uncertainty Estimation**: Measures confidence in generated outputs

### Educational Datasets Integration
- **Science QA Dataset**: Scientific educational content
- **Math Problem Datasets**: Mathematical reasoning and solutions
- **SQuAD Dataset**: Question-answering educational content
- **Wikipedia Educational Articles**: Curated educational topics
- **Curriculum Standards**: Common Core, STEM, and international standards

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd FYP

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### 2. Prepare Education Knowledge Base

```bash
# Generate education-specific knowledge base
python prepare_education_knowledge.py
```

This will create:
- `data/education_knowledge.txt` - Text-based knowledge base
- `data/education_knowledge.json` - Structured knowledge base
- `whoosh_index/` - Search index for fast retrieval

### 3. Run the System

```bash
# Start the Streamlit web interface
streamlit run app.py

# Or run tests
python test_education.py
```

## 📚 Educational Content Categories

The system is specialized for the following educational domains:

### 🧠 Pedagogy & Learning Theories
- Constructivism, Behaviorism, Cognitivism
- Social Learning Theory, Multiple Intelligences
- Bloom's Taxonomy, Learning Styles
- Differentiated Instruction, Project-Based Learning

### 📖 Curriculum & Standards
- Common Core State Standards
- International Baccalaureate (IB)
- Advanced Placement (AP)
- STEM/STEAM Education
- Special Education & Gifted Education

### 🎯 Subject Areas
- Mathematics Education
- Science Education
- Language Arts
- History & Geography
- Computer Science Education
- Arts & Music Education

### 🏫 Educational Systems
- K-12 Education Systems
- Higher Education
- International Education
- Distance Learning
- Educational Technology

### 📊 Assessment & Evaluation
- Formative & Summative Assessment
- Authentic Assessment
- Rubrics & Grading
- Portfolio Assessment
- Standardized Testing

## 🔧 Configuration

### Environment Variables
```bash
OPENAI_API_KEY=your_openai_api_key
GEMINI_API_KEY=your_gemini_api_key
```

### Model Selection
- **OpenAI Models**: `gpt-3.5-turbo` (default), `gpt-4`
- **Gemini Model**: `gemini-2.0-flash`
- **Local Models**: `EleutherAI/gpt-neo-125M` (fallback)

### Detection Parameters
- **Minimum Fact Similarity**: 0.1-0.9 (default: 0.3)
- **Consensus Threshold**: 0.3-0.9 (default: 0.5)
- **Knowledge Base Sources**: Education datasets, Wikipedia, curriculum standards

## 📊 Usage Examples

### Example 1: Pedagogical Content Verification
```python
query = "What is constructivism in education?"
output = "Constructivism is a learning theory that suggests students construct knowledge through experience and reflection."

# System will verify against educational knowledge base
# Check pedagogical accuracy, learning theory definitions
# Compare with established educational research
```

### Example 2: Curriculum Standard Verification
```python
query = "Explain the Common Core State Standards"
output = "Common Core standards are weekly tests that students must pass with 100% accuracy."

# System will flag this as hallucination
# Provide correct information about Common Core
# Reference official educational standards
```

### Example 3: STEM Education Content
```python
query = "What is STEM education?"
output = "STEM education integrates Science, Technology, Engineering, and Mathematics in an interdisciplinary approach."

# System will verify this accurate description
# Check against educational policy documents
# Validate interdisciplinary approach claims
```

## 🧪 Testing

### Run Comprehensive Tests
```bash
python test_education.py
```

### Test Categories
- **Correct Educational Content**: Should pass verification
- **Hallucinated Educational Claims**: Should be flagged
- **Knowledge Base Retrieval**: Test fact finding
- **Multi-Model Consensus**: Verify agreement between models

### Sample Test Cases
1. **Learning Theories**: Constructivism, Behaviorism, Cognitivism
2. **Educational Standards**: Common Core, IB, AP
3. **Teaching Methods**: Project-based learning, Flipped classroom
4. **Assessment**: Formative vs Summative assessment
5. **Educational Technology**: Online learning, Blended learning

## 📈 Performance Metrics

### Detection Accuracy
- **True Positives**: Correctly identified hallucinations
- **True Negatives**: Correctly identified accurate content
- **False Positives**: Incorrectly flagged accurate content
- **False Negatives**: Missed hallucinations

### Educational Specificity
- **Curriculum Alignment**: Accuracy with educational standards
- **Pedagogical Correctness**: Validation of teaching methods
- **Academic Rigor**: Scholarly accuracy of claims
- **Domain Relevance**: Educational content relevance

## 🔍 System Architecture

```
Education Knowledge Base
├── Curriculum Standards
├── Pedagogical Theories
├── Subject-Specific Content
├── Assessment Methods
└── Educational Systems

Detection Pipeline
├── Fact Retrieval (Hybrid)
├── Multi-Model Verification
├── Self-Consistency Check
├── Metamorphic Testing
└── Risk Aggregation

Output
├── Hallucination Detection
├── Educational Corrections
├── Explanations
└── Evidence
```

## 🛠️ Customization

### Adding New Educational Domains
1. Update `EDUCATION_TOPICS` in `prepare_education_knowledge.py`
2. Add domain-specific datasets
3. Update verification prompts for domain specificity
4. Test with domain-specific examples

### Extending Knowledge Base
```python
# Add new educational facts
def add_custom_education_facts():
    facts = [
        "Your custom educational fact here",
        "Another educational concept",
    ]
    return facts
```

### Custom Verification Prompts
```python
# Modify verification prompts for specific domains
def custom_education_verification_prompt(claim, facts):
    return f"""
    You are an expert in [specific domain] education.
    Verify this claim: {claim}
    Against these facts: {facts}
    Focus on [domain-specific criteria]
    """
```

## 📝 API Usage

### Basic Usage
```python
from retrieval import EducationFactRetriever
from verify import verify_claim_with_openai, verify_claim_with_gemini

# Initialize retriever
retriever = EducationFactRetriever()

# Retrieve relevant facts
facts = retriever.retrieve_facts("constructivism education", top_k=5)

# Verify claim
result = verify_claim_with_openai(claim, facts, api_key)
```

### Advanced Usage
```python
from detect import self_consistency_check, metamorphic_check
from flag import aggregate_risk

# Multiple detection methods
consistency_score = self_consistency_check(query, n=5)
meta_score = metamorphic_check(query)
risk_score = aggregate_risk(consistency_score, meta_score, ...)
```

## 🤝 Contributing

### Adding Educational Content
1. **Datasets**: Add new educational datasets
2. **Knowledge Base**: Expand educational facts
3. **Verification**: Improve domain-specific verification
4. **Testing**: Add domain-specific test cases

### Educational Expertise Areas
- **Curriculum Development**: Standards and frameworks
- **Pedagogy**: Teaching methods and theories
- **Assessment**: Evaluation and testing
- **Educational Technology**: Digital learning tools
- **Special Education**: Inclusive education practices

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Educational Datasets**: Science QA, GSM8K, SQuAD
- **Wikipedia**: Educational content and articles
- **OpenAI & Google**: LLM APIs for verification
- **Educational Research**: Pedagogical theories and practices

## 📞 Support

For questions about the education-specific system:
- **Educational Content**: Domain-specific verification
- **Technical Issues**: System functionality
- **Customization**: Domain adaptation
- **Performance**: Detection accuracy

---

**🎓 Built specifically for educational content verification and hallucination detection** 