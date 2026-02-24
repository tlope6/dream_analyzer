# """
# # NLP Pipeline — processes dream text to extract symbols, emotions, and embeddings.

# # Uses:
# # - spaCy for entity/noun extraction
# # - Custom dream symbol matching
# # - Zero-shot classification for emotion detection
# # - Sentence-transformers for semantic embeddings

# # For lightweight/offline mode (no large models), set USE_TRANSFORMERS=False.
# # """

import re
from collections import Counter

# ── Configuration ──────────────────────────────────────────────────────────
USE_TRANSFORMERS = True  # Set False for lightweight mode (no HuggingFace models)

# ── Lazy-loaded globals ────────────────────────────────────────────────────
_nlp = None
_emotion_classifier = None
_embedder = None


def _get_spacy():
    global _nlp
    if _nlp is None:
        import spacy
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading spaCy model...")
            from spacy.cli.download import download
            download("en_core_web_sm")
            _nlp = spacy.load("en_core_web_sm")
    return _nlp


def _get_emotion_classifier():
    global _emotion_classifier
    if _emotion_classifier is None and USE_TRANSFORMERS:
        from transformers import pipeline
        _emotion_classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=-1,  # CPU; change to 0 for GPU
        )
    return _emotion_classifier


def _get_embedder():
    global _embedder
    if _embedder is None and USE_TRANSFORMERS:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


#Emotion labels 
EMOTION_LABELS = [
    "joy",
    "fear",
    "anxiety",
    "sadness",
    "anger",
    "wonder",
    "confusion",
    "peace",
    "love",
    "power",
]


#  Symbol extraction
def extract_symbols(text: str) -> list[dict]:
    """
    Extract dream symbols using spaCy NER + noun extraction + dream dictionary.
    Returns list of {text, label, source} dicts.
    """
    from dream_sumbols import DREAM_SYMBOLS, SYMBOL_CATEGORIES

    nlp = _get_spacy()
    doc = nlp(text)
    symbols = []
    seen = set()
    text_lower = text.lower()

    # spaCy named entities (people, places, etc.)
    for ent in doc.ents:
        key = ent.text.lower()
        if key not in seen and ent.label_ in {"PERSON", "GPE", "LOC", "ORG", "FAC"}:
            symbols.append({
                "text": ent.text,
                "label": ent.label_,
                "source": "spacy_ner",
            })
            seen.add(key)

    # Dream symbol dictionary matching (multi-word first, then single-word)
    sorted_symbols = sorted(DREAM_SYMBOLS.keys(), key=len, reverse=True)
    for symbol_key in sorted_symbols:
        if symbol_key in text_lower and symbol_key not in seen:
            cat_key = DREAM_SYMBOLS[symbol_key]
            cat_name = SYMBOL_CATEGORIES.get(cat_key, cat_key)
            symbols.append({
                "text": symbol_key,
                "label": cat_name,
                "source": "dream_dictionary",
            })
            seen.add(symbol_key)

    # Significant nouns not already captured
    for token in doc:
        if (
            token.pos_ in ("NOUN", "PROPN")
            and token.text.lower() not in seen
            and len(token.text) > 2
            and not token.is_stop
        ):
            symbols.append({
                "text": token.text.lower(),
                "label": "noun",
                "source": "spacy_pos",
            })
            seen.add(token.text.lower())

    return symbols


# Emotion analysis 
def analyze_emotions(text: str) -> dict[str, float]:
    """
    Classify the emotional tone of a dream.
    Returns dict of {emotion: confidence_score}.
    Falls back to keyword-based if transformers unavailable.
    """
    classifier = _get_emotion_classifier()

    if classifier is not None:
        result = classifier(text, EMOTION_LABELS, multi_label=True)
        return {
            label: round(score, 3)
            for label, score in zip(result["labels"], result["scores"])
        }

    # Fallback: keyword-based emotion detection 
    return _keyword_emotion_analysis(text)


def _keyword_emotion_analysis(text: str) -> dict[str, float]:
    """Simple keyword-based emotion scoring as fallback."""
    emotion_keywords = {
        "joy": ["happy", "joy", "beautiful", "wonderful", "laughing", "smiled", "bright", "warm", "peaceful", "delight"],
        "fear": ["afraid", "scared", "terrified", "horror", "panic", "scream", "nightmare", "monster", "dark", "threatening"],
        "anxiety": ["nervous", "worried", "anxious", "stress", "pressure", "rush", "late", "exam", "test", "unprepared"],
        "sadness": ["sad", "crying", "tears", "loss", "grief", "lonely", "empty", "missing", "funeral", "gone"],
        "anger": ["angry", "furious", "rage", "fight", "yelling", "frustrated", "destroy", "attack", "violence"],
        "wonder": ["amazing", "strange", "bizarre", "magical", "surreal", "floating", "glowing", "impossible", "incredible"],
        "confusion": ["confused", "lost", "maze", "strange", "unclear", "fog", "didn't understand", "blank", "unfamiliar"],
        "peace": ["calm", "serene", "peaceful", "quiet", "gentle", "still", "relaxed", "comfort", "safe", "soft"],
        "love": ["love", "loved", "hug", "kiss", "embrace", "together", "partner", "family", "care", "tender"],
        "power": ["powerful", "strong", "control", "flying", "invincible", "mighty", "command", "dominant", "force"],
    }

    text_lower = text.lower()
    scores = {}
    for emotion, keywords in emotion_keywords.items():
        count = sum(1 for kw in keywords if kw in text_lower)
        scores[emotion] = round(min(count / 3, 1.0), 3)  # Normalize to 0-1

    # Normalize so they sum to ~1
    total = sum(scores.values())
    if total > 0:
        scores = {k: round(v / total, 3) for k, v in scores.items()}

    return scores


#  Embeddings 
def compute_embedding(text: str) -> list[float]:
    """Generate a semantic embedding vector for a dream text."""
    embedder = _get_embedder()
    if embedder is not None:
        vec = embedder.encode(text)
        return vec.tolist()
    return []


# Full pipeline 
def process_dream(text: str) -> dict:
    """
    Run the full NLP pipeline on a dream text.
    Returns dict with symbols, emotions, and embedding.
    """
    symbols = extract_symbols(text)
    emotions = analyze_emotions(text)
    embedding = compute_embedding(text)

    return {
        "symbols": symbols,
        "symbol_names": [s["text"] for s in symbols],
        "entities": [s for s in symbols if s["source"] == "spacy_ner"],
        "emotions": emotions,
        "embedding": embedding,
    }