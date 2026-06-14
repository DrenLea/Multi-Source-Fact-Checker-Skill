"""
Claim Processor Module

Preprocesses an incoming text claim: normalizes it, determines whether it is
fact-checkable, and decomposes compound claims into atomic sub-claims.
"""

import re
from typing import List

from loguru import logger


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Patterns that suggest a claim is NOT fact-checkable (subjective, future, etc.)
_SUBJECTIVE_PATTERNS: List[str] = [
    r"\bI (think|believe|feel|hope|wish|guess)\b",
    r"\bin my opinion\b",
    r"\b(should|ought to|must)\b",  # normative statements
    r"\b(best|worst|greatest|most beautiful)\b",  # pure opinion
    r"\b(will|going to) (be|become|happen|moon|skyrocket|crash|dump)\b",  # future prediction
    r"\bby (next year|202[7-9]|203)\b",  # far-future prediction
    r"\b(may|might|could possibly)\b",  # vague speculation (allow "may" only if paired with date)
]

# Minimum character length for a checkable claim
_MIN_CLAIM_LENGTH: int = 10

# Maximum character length before truncation warning
_MAX_CLAIM_LENGTH: int = 5000

# Regex to split compound sentences on common conjunction patterns
_DECOMPOSE_PATTERN: re.Pattern = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z])"  # split on sentence boundaries
)

# Additional splitting on explicit connectors within a long sentence
_CONNECTOR_SPLIT: re.Pattern = re.compile(
    r"\s+(?:and also|furthermore|moreover|additionally|in addition)\s*[,]?\s+",
    flags=re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def normalize_claim(text: str) -> str:
    """
    Normalize a claim string for consistent downstream processing.

    - Strips leading/trailing whitespace.
    - Collapses multiple spaces, tabs, and newlines into a single space.
    - Normalizes Unicode punctuation (curly quotes → straight quotes, etc.).

    Args:
        text: Raw input claim string.

    Returns:
        Normalized string.

    Examples:
        >>> normalize_claim('  Bitcoin   is   "digital   gold"  ')
        'Bitcoin is "digital gold"'
    """
    if not isinstance(text, str):
        raise TypeError(f"Expected str, got {type(text).__name__}")

    # Strip leading/trailing whitespace
    cleaned = text.strip()

    # Collapse multiple whitespace characters into a single space
    cleaned = re.sub(r"\s+", " ", cleaned)

    # Normalize common Unicode characters
    cleaned = cleaned.replace("‘", "'").replace("’", "'")  # curly single quotes
    cleaned = cleaned.replace("“", '"').replace("”", '"')  # curly double quotes
    cleaned = cleaned.replace("–", "--").replace("—", "---")  # en/em dashes

    logger.debug(f"Normalized claim: {cleaned[:80]}...")
    return cleaned


def is_checkable(claim: str) -> bool:
    """
    Determine whether a claim is suitable for fact-checking.

    A claim is NOT checkable if it is:
    - A subjective opinion ("I think...")
    - A normative statement ("should...")
    - A far-future prediction
    - Too short to be meaningful
    - A question rather than an assertion

    Args:
        claim: Normalized claim text.

    Returns:
        True if the claim can be fact-checked, False otherwise.
    """
    claim = claim.strip()

    # Reject empty or very short strings
    if len(claim) < _MIN_CLAIM_LENGTH:
        logger.debug("Claim too short to be checkable")
        return False

    # Reject questions (ends with '?' and typical question words)
    if claim.endswith("?"):
        logger.debug("Claim is a question, not an assertion")
        return False

    # Check against subjective/prediction patterns
    for pattern in _SUBJECTIVE_PATTERNS:
        if re.search(pattern, claim, flags=re.IGNORECASE):
            logger.debug(f"Claim matched non-checkable pattern: {pattern}")
            return False

    return True


def decompose_claims(text: str) -> List[str]:
    """
    Decompose a potentially compound claim into atomic sub-claims.

    Strategy (layered):
    1. Split on sentence boundaries (period, exclamation, question mark).
    2. Further split long sentences on explicit connector words.
    3. Return the list of atomic claims, each normalized.

    If no decomposition is possible, returns a single-element list containing
    the original text.

    Args:
        text: Normalized claim text.

    Returns:
        List of atomic claim strings.
    """
    # Step 1: Split on sentence boundaries
    sentences = _DECOMPOSE_PATTERN.split(text)

    # Step 2: Further split long sentences on connectors
    atomic: List[str] = []
    for sentence in sentences:
        sentence = sentence.strip().rstrip(".")
        if not sentence:
            continue
        # Attempt connector split
        parts = _CONNECTOR_SPLIT.split(sentence)
        for part in parts:
            part = part.strip().strip(",").strip()
            if part and len(part) >= _MIN_CLAIM_LENGTH:
                atomic.append(part)

    # If nothing useful emerged, return the original
    if not atomic:
        atomic.append(text)

    logger.info(f"Decomposed claim into {len(atomic)} atomic claim(s)")
    for i, c in enumerate(atomic):
        logger.debug(f"  [{i}] {c}")

    return atomic


# ---------------------------------------------------------------------------
# Mock Paladin-mini claim decomposition (placeholder for ML-based decomposition)
# ---------------------------------------------------------------------------

def decompose_claims_ml(text: str) -> List[str]:
    """
    ML-enhanced claim decomposition using a lightweight model.

    NOTE: Currently falls back to rule-based decomposition. To use the actual
    Paladin-mini or Loki claim_decomposer, set up the model path in
    config/api_config.yaml and uncomment the relevant code below.

    Args:
        text: Normalized claim text.

    Returns:
        List of atomic claim strings.
    """
    # Attempt to use Loki's claim_decomposer if available
    try:
        # import torch
        # from loki.claim_decomposer import ClaimDecomposer
        # model = ClaimDecomposer.from_pretrained("princeton-nlp/loki-base")
        # return model.decompose(text)
        raise ImportError("Loki not installed — falling back to rule-based decomposition")
    except (ImportError, Exception) as exc:
        logger.debug(f"ML decomposition unavailable ({exc}), using rule-based fallback")
        return decompose_claims(text)
