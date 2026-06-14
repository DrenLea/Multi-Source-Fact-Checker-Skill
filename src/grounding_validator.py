"""
Grounding Validator Module

Validates how well the retrieved evidence supports or contradicts the claim
by computing a grounding consistency score.

Primary method: Paladin-mini (3.8B parameter) compact inference model.
Fallback: Keyword overlap heuristic when the model is unavailable.
"""

import re
from typing import Dict, List, Optional

from loguru import logger

from .utils import load_yaml_config, resolve_env_vars


# ---------------------------------------------------------------------------
# Model Cache (module-level singleton)
# ---------------------------------------------------------------------------

_model = None
_tokenizer = None
_model_attempted: bool = False  # Track whether we already tried to load the model


def _load_paladin_mini():
    """
    Lazy-load the Paladin-mini model and tokenizer from HuggingFace.

    The model is downloaded on first use and cached for subsequent calls.
    Falls back gracefully if the model cannot be loaded.
    """
    global _model, _tokenizer, _model_attempted

    if _model_attempted:
        return _model, _tokenizer

    _model_attempted = True

    cfg = load_yaml_config("api_config.yaml")
    cfg = resolve_env_vars(cfg)
    model_path = cfg.get("loki", {}).get("model_path", "princeton-nlp/loki-base")
    device = cfg.get("loki", {}).get("device", "cpu")

    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        logger.info(f"Loading Paladin-mini model from {model_path} on {device}...")
        _tokenizer = AutoTokenizer.from_pretrained(model_path)
        _model = AutoModelForSequenceClassification.from_pretrained(model_path)
        _model.to(device)
        _model.eval()
        logger.info("Paladin-mini model loaded successfully")
    except ImportError:
        logger.warning(
            "transformers library not installed — grounding validation will use keyword fallback. "
            "Install with: pip install transformers torch"
        )
    except Exception as exc:
        logger.warning(
            f"Failed to load Paladin-mini model: {exc} — "
            "grounding validation will use keyword overlap fallback"
        )

    return _model, _tokenizer


# ---------------------------------------------------------------------------
# ML-based Grounding Check
# ---------------------------------------------------------------------------

def _grounding_ml(claim: str, evidence_text: str) -> Optional[float]:
    """
    Compute grounding score using the Paladin-mini model.

    Args:
        claim: The atomic claim text.
        evidence_text: Concatenated evidence passages.

    Returns:
        Score in [0, 1] or None if model is unavailable.
    """
    model, tokenizer = _load_paladin_mini()

    if model is None or tokenizer is None:
        return None

    try:
        import torch

        # Concatenate claim and evidence as input pair
        inputs = tokenizer(
            claim,
            evidence_text,
            max_length=512,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        # Move to the same device as the model
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            # Assume binary classification: class 1 = grounded/supported
            probs = torch.softmax(logits, dim=-1)
            grounding_score = probs[0, 1].item() if probs.shape[-1] >= 2 else probs[0, 0].item()

        return round(float(grounding_score), 4)
    except Exception as exc:
        logger.error(f"Error during ML grounding inference: {exc}")
        return None


# ---------------------------------------------------------------------------
# Keyword-Overlap Fallback
# ---------------------------------------------------------------------------

def _grounding_keywords(claim: str, evidence_text: str) -> float:
    """
    Compute a simple grounding score based on keyword overlap.

    This is a heuristic fallback that:
    1. Extracts meaningful tokens (3+ chars, non-stopword) from the claim.
    2. Checks how many appear in the evidence text.
    3. Returns the overlap ratio.

    Args:
        claim: The atomic claim text.
        evidence_text: Concatenated evidence passages.

    Returns:
        Score in [0, 1].
    """
    # Simple English stopwords list
    _STOPWORDS: set = {
        "the", "is", "at", "which", "on", "a", "an", "and", "or", "but",
        "in", "with", "to", "for", "of", "from", "by", "as", "be", "was",
        "are", "been", "being", "have", "has", "had", "do", "does", "did",
        "will", "would", "could", "should", "may", "might", "shall", "can",
        "not", "no", "nor", "so", "if", "than", "that", "this", "these",
        "those", "it", "its", "they", "them", "their",
    }

    def _tokenize(text: str) -> List[str]:
        """Extract lowercase alphanumeric tokens, filtering stopwords and short tokens."""
        tokens = re.findall(r"[a-zA-Z0-9]{3,}", text.lower())
        return [t for t in tokens if t not in _STOPWORDS]

    claim_tokens = set(_tokenize(claim))
    evidence_tokens = set(_tokenize(evidence_text))

    if not claim_tokens:
        logger.debug("No meaningful tokens in claim — returning neutral 0.5")
        return 0.5

    overlap = claim_tokens & evidence_tokens
    ratio = len(overlap) / len(claim_tokens)

    logger.debug(
        f"Keyword grounding: {len(overlap)}/{len(claim_tokens)} tokens overlap → {ratio:.3f}"
    )
    return round(ratio, 4)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_grounding(claim: str, evidence_text: str) -> float:
    """
    Validate how well the evidence supports the claim.

    Tries ML-based grounding first (Paladin-mini), then falls back to
    keyword overlap.

    Args:
        claim: The atomic claim text.
        evidence_text: Concatenated evidence passages from all sources.

    Returns:
        Grounding consistency score in [0, 1].
        - 1.0: Evidence strongly supports the claim.
        - 0.0: Evidence strongly contradicts the claim.
        - 0.5: No clear signal either way.
    """
    if not evidence_text or not evidence_text.strip():
        logger.warning("Empty evidence text — returning neutral grounding score 0.5")
        return 0.5

    # Try ML grounding
    ml_score = _grounding_ml(claim, evidence_text)
    if ml_score is not None:
        logger.info(f"ML grounding score: {ml_score:.4f}")
        return ml_score

    # Fallback to keyword overlap
    logger.info("Using keyword-overlap grounding fallback")
    return _grounding_keywords(claim, evidence_text)


def build_evidence_text(all_evidences: dict) -> str:
    """
    Concatenate evidence from all sources into a single text for grounding validation.

    Args:
        all_evidences: Evidence dict from evidence_retriever.retrieve_all_evidence().

    Returns:
        Concatenated text string.
    """
    parts: List[str] = []

    # Web2 evidence excerpts
    web2 = all_evidences.get("web2", {})
    details = web2.get("details", {})
    for sub_key in ["google_fact_check", "loki_evidence"]:
        sub = details.get(sub_key, {})
        passages = sub.get("details", {}).get("top_passage", "")
        if passages:
            parts.append(str(passages))
        textual_rating = sub.get("details", {}).get("textual_rating", "")
        if textual_rating:
            parts.append(f"Fact-check rating: {textual_rating}")

    # Social graph signals
    social = all_evidences.get("web3_social", {})
    parts.append(
        f"Social propagation: {social.get('propagation_pattern', 'unknown')}, "
        f"reposts: {social.get('repost_count', 0)}, "
        f"first-poster reputation: {social.get('first_poster_reputation', 0.5)}"
    )

    # Chain reputation signals
    chain = all_evidences.get("chain_reputation", {})
    parts.append(
        f"On-chain reputation: zScore={chain.get('zscore', 50)}, "
        f"aura={chain.get('aura', 50)}, "
        f"payment_backing={chain.get('payment_backing', 0.0)}"
    )

    combined = " | ".join(parts)
    return combined
