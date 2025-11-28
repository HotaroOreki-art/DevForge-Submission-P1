import os
import re
import nltk
import traceback
from typing import List, Dict

# Add vendored nltk_data directory to nltk search path so deployments
# can ship the required corpora inside `env/nltk_data`.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
VENDORED_NLTK = os.path.join(PROJECT_ROOT, 'env', 'nltk_data')
if os.path.isdir(VENDORED_NLTK):
    # Put vendored path first so it takes precedence
    nltk.data.path.insert(0, VENDORED_NLTK)
else:
    # Ensure directory exists for future vendoring
    try:
        os.makedirs(VENDORED_NLTK, exist_ok=True)
        nltk.data.path.insert(0, VENDORED_NLTK)
    except Exception:
        pass

# Try to ensure required resources exist (best-effort). If downloads fail
# we silently fall back to the lightweight extractor below.
def _ensure_nltk_resource(name: str, download_name: str = None):
    try:
        nltk.data.find(name)
        return True
    except LookupError:
        try:
            nltk.download(download_name or name.split('/')[-1])
            return True
        except Exception:
            return False

# Required resources
_have_punkt = _ensure_nltk_resource('tokenizers/punkt', 'punkt')
_have_perceptron = _ensure_nltk_resource('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger')

USE_NLTK = _have_punkt and _have_perceptron

"""
This module extracts simple (subject, relation, object) triples
from input text.

Behaviour:
- If NLTK resources are available (vendored or downloadable) we use
  sentence tokenization and POS tagging for higher-quality triples.
- If resources are not available, a lightweight regex-based fallback
  extractor is used so that the server works offline.
"""

RELATION_VERBS = [
    "is", "are", "was", "were",
    "has", "have",
    "belongs", "belong",
    "created", "creates",
    "discovered", "discovers",
    "contains", "contain",
    "includes", "include",
    "uses", "used",
    "makes", "made",
    "forms", "formed",
    "evolves", "evolve",
    "wins", "won",
    "leads", "led",
    "developed", "develops",
    "produced", "produces",
    "invented", "invents",
    "studied", "studies"
]


# --- NLTK-powered extractor ---
if USE_NLTK:
    from nltk import pos_tag, word_tokenize, sent_tokenize

    def _extract_with_nltk(text: str) -> List[Dict]:
        sentences = sent_tokenize(text)
        triples = []

        for sent in sentences:
            tokens = word_tokenize(sent)
            try:
                tagged = pos_tag(tokens)
            except Exception:
                # If tagging fails for some reason, skip this sentence
                continue

            # Find simple subject, verb, object patterns
            for i in range(len(tagged) - 2):
                w1, t1 = tagged[i]
                w2, t2 = tagged[i+1]
                w3, t3 = tagged[i+2]

                # Pattern 1: Noun - Verb - Noun
                if t1.startswith("NN") and t2.startswith("VB") and t3.startswith("NN"):
                    triples.append({
                        "subject": w1,
                        "relation": w2.lower(),
                        "object": w3
                    })

                # Pattern 2: Noun - "is/are/was/were" - Adjective/Noun
                if t1.startswith("NN") and w2.lower() in ["is", "are", "was", "were"] and (t3.startswith("JJ") or t3.startswith("NN")):
                    triples.append({
                        "subject": w1,
                        "relation": w2.lower(),
                        "object": w3
                    })

                # Pattern 3: Known relation verbs list
                if t1.startswith("NN") and w2.lower() in RELATION_VERBS and t3.startswith(("NN", "JJ")):
                    triples.append({
                        "subject": w1,
                        "relation": w2.lower(),
                        "object": w3
                    })

        return triples

else:
    # --- Lightweight regex-based fallback extractor ---
    # This is intentionally simple and conservative so it works offline.
    _SIMPLE_PATTERNS = [
        # "X is a Y"
        re.compile(r"(?P<subject>\b[A-Z][a-zA-Z0-9_-]+)\s+(is|are|was|were)\s+(a|an|the)?\s*(?P<object>[A-Za-z0-9_ -]+)", re.IGNORECASE),
        # "X evolves into Y" / "X evolves to Y"
        re.compile(r"(?P<subject>\b[A-Z][a-zA-Z0-9_-]+)\s+evolves\s+(into|to)\s+(?P<object>[A-Za-z0-9_ -]+)", re.IGNORECASE),
        # simple "X has Y"
        re.compile(r"(?P<subject>\b[A-Z][a-zA-Z0-9_-]+)\s+has\s+(?P<object>[A-Za-z0-9_ -]+)", re.IGNORECASE),
    ]

    def _extract_fallback(text: str) -> List[Dict]:
        triples = []
        # Split into sentence-like chunks by punctuation
        parts = re.split(r"[\.\n;!\?]+", text)
        for part in parts:
            part = part.strip()
            if not part:
                continue
            for pat in _SIMPLE_PATTERNS:
                m = pat.search(part)
                if not m:
                    continue
                subj = m.groupdict().get('subject') or ''
                obj = m.groupdict().get('object') or ''
                rel = 'related_to'
                # choose relation from words present if possible
                if 'evolv' in part.lower():
                    rel = 'evolves_into'
                elif 'is' in part.lower() or 'are' in part.lower():
                    rel = 'is'
                elif 'has' in part.lower():
                    rel = 'has'

                triples.append({
                    'subject': subj.strip(),
                    'relation': rel,
                    'object': obj.strip()
                })
        return triples


# Public API
def extract_relations(text: str) -> List[Dict]:
    """Return list of triples: {subject, relation, object}.

    Uses the NLTK extractor when resources are available, otherwise
    falls back to a conservative regex-based extractor.
    """
    try:
        if USE_NLTK:
            return _extract_with_nltk(text)
        else:
            return _extract_fallback(text)
    except Exception:
        # If anything unexpected fails, return empty list rather than
        # crashing the application.
        traceback.print_exc()
        return []
