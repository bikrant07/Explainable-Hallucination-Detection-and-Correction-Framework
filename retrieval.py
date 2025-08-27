"""
retrieval.py
Implements hybrid fact retrieval and cross-checking for hallucination detection.
Uses FAISS for dense vector retrieval and Whoosh for sparse keyword-based search.
Combines results for best recall and accuracy.
"""
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Dict
from whoosh.index import create_in
from whoosh.fields import *
from whoosh.qparser import QueryParser
from whoosh.analysis import StandardAnalyzer
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
import re

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')
try:
    nltk.data.find('chunkers/maxent_ne_chunker')
except LookupError:
    nltk.download('maxent_ne_chunker')

FACTS_PATH = os.path.join("data", "education_knowledge.txt")
EMBED_MODEL = "all-MiniLM-L6-v2"
INDEX_DIR = "whoosh_index"

class EducationFactRetriever:
    """
    Education-specific fact retriever combining dense (FAISS) and sparse (Whoosh) retrieval.
    Specialized for educational content, curriculum, and pedagogical facts.
    """
    def __init__(self, facts_path=FACTS_PATH, embed_model=EMBED_MODEL, use_wikipedia=False, wiki_top_k=3):
        self.facts = self._load_facts(facts_path)
        self.embedder = SentenceTransformer(embed_model)
        self.use_wikipedia = use_wikipedia
        self.wiki_top_k = wiki_top_k
        
        # Initialize dense retriever (FAISS)
        self.embeddings = self.embedder.encode(self.facts, convert_to_numpy=True)
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(self.embeddings)
        
        # Initialize sparse retriever (Whoosh)
        self._setup_whoosh_index()
        
        if use_wikipedia:
            try:
                import wikipedia
                self.wikipedia = wikipedia
            except ImportError:
                self.wikipedia = None
                print("Wikipedia package not installed. Wikipedia retrieval disabled.")

    def _load_facts(self, path) -> List[str]:
        """Load facts from file."""
        with open(path, "r", encoding="utf-8") as f:
            facts = [line.strip() for line in f if line.strip()]
        return facts

    def _setup_whoosh_index(self):
        """Setup Whoosh index for sparse retrieval."""
        if not os.path.exists(INDEX_DIR):
            os.mkdir(INDEX_DIR)
        
        schema = Schema(content=TEXT(stored=True), fact_id=ID(stored=True))
        self.ix = create_in(INDEX_DIR, schema)
        writer = self.ix.writer()
        
        for i, fact in enumerate(self.facts):
            writer.add_document(content=fact, fact_id=str(i))
        writer.commit()

    def _extract_keywords(self, query: str) -> List[str]:
        """Extract keywords and named entities from query."""
        try:
            # Tokenize and remove stopwords
            stop_words = set(stopwords.words('english'))
            tokens = word_tokenize(query.lower())
            keywords = [token for token in tokens if token.isalnum() and token not in stop_words]
            
            # Extract named entities (with error handling)
            try:
                pos_tags = pos_tag(word_tokenize(query))
                named_entities = ne_chunk(pos_tags)
                entities = []
                for chunk in named_entities:
                    if hasattr(chunk, 'label'):
                        entities.append(' '.join([token for token, pos in chunk.leaves()]))
                keywords.extend(entities)
            except Exception as e:
                # Fallback: extract capitalized words as potential entities
                capitalized_words = re.findall(r'\b[A-Z][a-z]+\b', query)
                keywords.extend(capitalized_words)
            
            return keywords
        except Exception as e:
            # Ultimate fallback: simple word extraction
            words = re.findall(r'\b\w+\b', query.lower())
            stop_words = set(stopwords.words('english'))
            return [word for word in words if word not in stop_words and len(word) > 2]

    def _dense_retrieve(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """Dense retrieval using FAISS."""
        q_emb = self.embedder.encode([query], convert_to_numpy=True)
        D, I = self.index.search(q_emb, k)
        results = []
        for i, (dist, idx) in enumerate(zip(D[0], I[0])):
            if idx < len(self.facts):
                similarity = 1.0 / (1.0 + dist)  # Convert distance to similarity
                results.append((self.facts[idx], similarity))
        return results

    def _sparse_retrieve(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """Sparse retrieval using Whoosh."""
        with self.ix.searcher() as searcher:
            qp = QueryParser("content", self.ix.schema)
            q = qp.parse(query)
            results = searcher.search(q, limit=k)
            
            retrieved = []
            for result in results:
                fact = result['content']
                score = result.score
                retrieved.append((fact, score))
            return retrieved

    def _get_wikipedia_facts(self, query: str, top_k: int = 3) -> List[str]:
        """Fetch Wikipedia facts."""
        if self.wikipedia is None:
            return []
        try:
            search_results = self.wikipedia.search(query, results=top_k)
            facts = []
            for title in search_results:
                try:
                    summary = self.wikipedia.summary(title, sentences=2)
                    facts.append(summary)
                except Exception:
                    continue
            return facts
        except Exception:
            return []

    def retrieve(self, query: str, k: int = 5, min_similarity: float = 0.3) -> List[str]:
        """
        Hybrid retrieval combining dense and sparse search.
        Returns deduplicated, reranked facts above similarity threshold.
        """
        # Extract keywords for better retrieval
        keywords = self._extract_keywords(query)
        
        # Dense retrieval
        dense_results = self._dense_retrieve(query, k=k*2)
        
        # Sparse retrieval with original query and keywords
        sparse_results = self._sparse_retrieve(query, k=k)
        for keyword in keywords[:3]:  # Use top 3 keywords
            sparse_results.extend(self._sparse_retrieve(keyword, k=k//2))
        
        # Combine and deduplicate
        all_results = {}
        for fact, score in dense_results + sparse_results:
            if fact not in all_results:
                all_results[fact] = score
            else:
                all_results[fact] = max(all_results[fact], score)
        
        # Rerank by semantic similarity to query
        if all_results:
            facts = list(all_results.keys())
            fact_embeddings = self.embedder.encode(facts, convert_to_numpy=True)
            query_embedding = self.embedder.encode([query], convert_to_numpy=True)
            
            similarities = np.dot(fact_embeddings, query_embedding.T).flatten()
            
            # Combine original scores with semantic similarity
            reranked = []
            for i, fact in enumerate(facts):
                combined_score = (all_results[fact] + similarities[i]) / 2
                if combined_score >= min_similarity:
                    reranked.append((fact, combined_score))
            
            # Sort by combined score
            reranked.sort(key=lambda x: x[1], reverse=True)
            facts = [fact for fact, score in reranked[:k]]
        else:
            facts = []
        
        # Add Wikipedia facts if enabled
        if self.use_wikipedia and self.wikipedia is not None:
            wiki_facts = self._get_wikipedia_facts(query, self.wiki_top_k)
            facts.extend(wiki_facts)
        
        return facts

    def add_fact(self, fact: str):
        """Add a new fact to both dense and sparse indices."""
        # Add to facts list
        self.facts.append(fact)
        
        # Update dense index
        fact_emb = self.embedder.encode([fact], convert_to_numpy=True)
        self.index.add(fact_emb)
        
        # Update sparse index
        writer = self.ix.writer()
        writer.add_document(content=fact, fact_id=str(len(self.facts) - 1))
        writer.commit()

def cross_check(output: str, facts: List[str], threshold: float = 0.7) -> List[Tuple[str, float]]:
    """
    For each sentence in output, check semantic similarity to retrieved facts.
    Returns list of (sentence, max_similarity) for sentences with low similarity (potential hallucinations).
    """
    from sentence_transformers import util
    sentences = sent_tokenize(output)
    embedder = SentenceTransformer(EMBED_MODEL)
    sent_embs = embedder.encode(sentences, convert_to_numpy=True)
    fact_embs = embedder.encode(facts, convert_to_numpy=True)
    flagged = []
    for i, sent in enumerate(sentences):
        sims = util.cos_sim(sent_embs[i], fact_embs)[0].cpu().numpy()
        max_sim = float(np.max(sims))
        if max_sim < threshold:
            flagged.append((sent, max_sim))
    return flagged

# Backward compatibility
class FactRetriever(EducationFactRetriever):
    """Backward compatibility wrapper."""
    pass

# Example usage (for testing):
if __name__ == "__main__":
    retriever = EducationFactRetriever(use_wikipedia=True)
    query = "What is constructivism in education?"
    facts = retriever.retrieve(query, k=3)
    print("Top facts:", facts)
    output = "Constructivism is a learning theory that suggests students construct knowledge through experience and reflection."
    flagged = cross_check(output, facts)
    print("Flagged:", flagged)
