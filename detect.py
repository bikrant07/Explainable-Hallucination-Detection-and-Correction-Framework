"""
detect.py
Implements LLM output generation, self-consistency, uncertainty estimation, and metamorphic testing for education-specific hallucination detection.
Uses only open-source models and runs locally, but can also use OpenAI API if enabled.
Specialized for educational content, curriculum, and pedagogical discussions.
"""
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
import numpy as np
from scipy.stats import entropy
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Optional
from sentence_transformers import SentenceTransformer  # Ensure this is always imported

# Default model: small, open, runs on CPU/GPU. Swap model_name for larger models if resources allow.
# Using education-tuned models when available
MODEL_NAME = "EleutherAI/gpt-neo-125M"  # e.g., "EleutherAI/gpt-neo-1.3B" for bigger
PARAPHRASE_MODEL = "ramsrigouthamg/t5_paraphraser"  # More compatible T5-based paraphraser

# --- OpenAI API Support ---
def openai_available():
    try:
        import openai
        return True
    except ImportError:
        return False

def set_openai_key(api_key: Optional[str] = None):
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    elif "OPENAI_API_KEY" not in os.environ:
        raise ValueError("OpenAI API key must be set via argument or environment variable.")

# --- LLM Generation ---
def load_llm(model_name=MODEL_NAME):
    """Load a causal language model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    return model, tokenizer

def generate_output(prompt: str, model=None, tokenizer=None, max_new_tokens=64, return_probs=True, use_openai=False, openai_model="gpt-3.5-turbo", api_key=None) -> Tuple[str, List[float]]:
    """
    Generate a single completion for the prompt. Optionally return token probabilities.
    If use_openai is True, use OpenAI API; else use local model.
    """
    if use_openai:
        if not openai_available():
            raise ImportError("openai package not installed.")
        set_openai_key(api_key)
        import openai
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=openai_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=max_new_tokens,
        )
        text = response.choices[0].message.content.strip()
        # OpenAI API does not return per-token probabilities for chat models, so we can't get entropy directly
        return text, None
    # Local model fallback
    if model is None or tokenizer is None:
        model, tokenizer = load_llm()
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            output_scores=return_probs,
            return_dict_in_generate=True
        )
    generated_ids = output.sequences[0][inputs['input_ids'].shape[-1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    probs = None
    if return_probs and hasattr(output, 'scores') and output.scores:
        # Get softmax probabilities for each generated token
        probs = [torch.softmax(score[0], dim=0).max().item() for score in output.scores]
    return generated_text, probs

def self_consistency_check(prompt: str, n: int = 5, model=None, tokenizer=None, use_openai=False, openai_model="gpt-3.5-turbo", api_key=None) -> Tuple[List[str], float]:
    """
    Generate N completions and compute average pairwise cosine similarity (semantic consistency).
    Returns (generations, mean_similarity).
    """
    generations = []
    for _ in range(n):
        gen, _ = generate_output(prompt, model, tokenizer, use_openai=use_openai, openai_model=openai_model, api_key=api_key)
        generations.append(gen)
    # Embed generations using a sentence transformer for semantic similarity
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embedder.encode(generations)
    sim_matrix = cosine_similarity(embeddings)
    # Take upper triangle mean (excluding diagonal)
    n = len(generations)
    mean_sim = (np.sum(np.triu(sim_matrix, 1)) * 2) / (n * (n - 1))
    return generations, mean_sim

def uncertainty_score(text: str, probs: List[float]) -> Optional[float]:
    """
    Compute an entropy-like uncertainty score from token probabilities.
    Lower max-prob means higher uncertainty. Returns mean entropy.
    """
    if probs is None or len(probs) == 0:
        return None
    # Convert max-probs to pseudo-entropy: H = -log(p)
    entropies = [-np.log(p + 1e-8) for p in probs]
    return float(np.mean(entropies))

# --- Paraphrasing ---
def load_paraphraser(model_name=PARAPHRASE_MODEL):
    """Load a T5-based paraphrase model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.eval()
    return model, tokenizer

def paraphrase(prompt: str, n: int = 3, model=None, tokenizer=None, use_openai=False, openai_model="gpt-3.5-turbo", api_key=None) -> List[str]:
    """
    Generate n paraphrased versions of the prompt using a local T5 model or OpenAI API.
    """
    if use_openai:
        if not openai_available():
            raise ImportError("openai package not installed.")
        set_openai_key(api_key)
        import openai
        client = openai.OpenAI(api_key=api_key)
        paraphrases = []
        for _ in range(n):
            response = client.chat.completions.create(
                model=openai_model,
                messages=[{"role": "user", "content": f"Paraphrase the following: {prompt}"}],
                temperature=1.2,
                max_tokens=64,
            )
            text = response.choices[0].message.content.strip()
            paraphrases.append(text)
        return paraphrases
    # Local model fallback
    if model is None or tokenizer is None:
        model, tokenizer = load_paraphraser()
    input_text = f"paraphrase: {prompt} </s>"
    inputs = tokenizer([input_text]*n, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=64,
            num_beams=5,
            num_return_sequences=n,
            temperature=1.5
        )
    return [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]

def metamorphic_check(prompt: str, model=None, tokenizer=None, para_model=None, para_tokenizer=None, use_openai=False, openai_model="gpt-3.5-turbo", api_key=None) -> float:
    """
    Generate paraphrased prompts, get LLM outputs, and compute output variance (semantic drift).
    Returns mean pairwise similarity between outputs (lower = less stable).
    """
    paraphrases = paraphrase(prompt, n=3, model=para_model, tokenizer=para_tokenizer, use_openai=use_openai, openai_model=openai_model, api_key=api_key)
    outputs = []
    for p in paraphrases:
        out, _ = generate_output(p, model, tokenizer, use_openai=use_openai, openai_model=openai_model, api_key=api_key)
        outputs.append(out)
    # Semantic similarity
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embedder.encode(outputs)
    sim_matrix = cosine_similarity(embeddings)
    n = len(outputs)
    mean_sim = (np.sum(np.triu(sim_matrix, 1)) * 2) / (n * (n - 1))
    return mean_sim

# --- OpenAI Fact-Checking ---
def openai_fact_check(output: str, facts: List[str], api_key=None, openai_model="gpt-3.5-turbo") -> List[str]:
    """
    Use OpenAI API to check if output is consistent with provided facts. Returns list of flagged/conflicting statements.
    """
    if not openai_available():
        raise ImportError("openai package not installed.")
    set_openai_key(api_key)
    import openai
    client = openai.OpenAI(api_key=api_key)
    prompt = (
        "Given the following output and facts, identify any statements in the output that contradict the facts. "
        "List the conflicting statements only.\n\n"
        f"Output: {output}\n\nFacts: {facts}\n\nConflicting statements:"
    )
    response = client.chat.completions.create(
        model=openai_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=128,
    )
    flagged = response.choices[0].message.content.strip().split("\n")
    flagged = [s for s in flagged if s.strip()]
    return flagged

def generate_multiple_openai_completions(prompt: str, n: int = 3, api_key=None, openai_model="gpt-3.5-turbo") -> list:
    """
    Generate n completions from OpenAI for consensus checking.
    """
    import openai
    client = openai.OpenAI(api_key=api_key)
    completions = []
    for _ in range(n):
        response = client.chat.completions.create(
            model=openai_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=1.0,
            max_tokens=128,
        )
        completions.append(response.choices[0].message.content.strip())
    return completions

# --- Live Search (DuckDuckGo) ---
def fetch_duckduckgo_snippets(query: str, max_results: int = 2) -> list:
    """
    Fetch top search snippets from DuckDuckGo for the query.
    """
    try:
        from duckduckgo_search import DDGS
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                if 'body' in r:
                    results.append(r['body'])
                elif 'snippet' in r:
                    results.append(r['snippet'])
        return results
    except Exception as e:
        return []

# --- Consensus Check ---
def consensus_check(user_output: str, completions: list, search_snippets: list, threshold: float = 0.5) -> dict:
    """
    Compare user output to LLM completions and search snippets using semantic similarity.
    Uses max similarity for agreement (less strict).
    Returns dict with agreement flags and similarity scores.
    """
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    all_refs = completions + search_snippets
    if not all_refs:
        return {"llm_agree": False, "search_agree": False, "llm_sim": 0.0, "search_sim": 0.0}
    user_emb = embedder.encode([user_output])[0]
    ref_embs = embedder.encode(all_refs)
    sims = cosine_similarity([user_emb], ref_embs)[0]
    # LLM consensus: max similarity to completions
    llm_sim = float(np.max(sims[:len(completions)])) if completions else 0.0
    # Search consensus: max similarity to search snippets
    search_sim = float(np.max(sims[len(completions):])) if search_snippets else 0.0
    llm_agree = llm_sim >= threshold
    search_agree = search_sim >= threshold
    return {
        "llm_agree": llm_agree,
        "search_agree": search_agree,
        "llm_sim": llm_sim,
        "search_sim": search_sim
    }

# Example usage (for testing):
if __name__ == "__main__":
    import getpass
    api_key = os.environ.get("OPENAI_API_KEY") or getpass.getpass("OpenAI API key (optional): ")
    model, tokenizer = load_llm()
    prompt = "What is the capital of France?"
    out, probs = generate_output(prompt, model, tokenizer)
    print("Generated (local):", out)
    print("Uncertainty score:", uncertainty_score(out, probs))
    gens, sim = self_consistency_check(prompt, 3, model, tokenizer)
    print("Self-consistency:", sim)
    para_model, para_tokenizer = load_paraphraser()
    meta_sim = metamorphic_check(prompt, model, tokenizer, para_model, para_tokenizer)
    print("Metamorphic stability:", meta_sim)
    # OpenAI generation
    if openai_available() and api_key:
        out2, _ = generate_output(prompt, use_openai=True, api_key=api_key)
        print("Generated (OpenAI):", out2)
        gens2, sim2 = self_consistency_check(prompt, 3, use_openai=True, api_key=api_key)
        print("Self-consistency (OpenAI):", sim2)
        meta_sim2 = metamorphic_check(prompt, use_openai=True, api_key=api_key)
        print("Metamorphic stability (OpenAI):", meta_sim2)
