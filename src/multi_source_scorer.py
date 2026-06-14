"""
Multi-Source Scoring Module

Fuses evidence from multiple independent sources into a single credibility
rating. Implements dynamic weight adjustment, conflict detection, and
threshold-based rating classification.
"""

from typing import Dict, List, Optional, Tuple

from loguru import logger

from .utils import load_yaml_config


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Map source verdict strings to numeric scores
_VERDICT_SCORE_MAP: Dict[str, float] = {
    "SUPPORTS": 1.0,
    "LIKELY_SUPPORTS": 0.75,
    "MIXED": 0.5,
    "LIKELY_CONTRADICTS": 0.25,
    "CONTRADICTS": 0.0,
    "NO_EVIDENCE": 0.5,
    "UNKNOWN": 0.5,
    "TIMEOUT": 0.5,
    "ERROR": 0.5,
}

# Web2 sub-source verdict mapping (used when extracting from detailed web2 results)
_WEB2_SUB_VERDICT_MAP: Dict[str, float] = _VERDICT_SCORE_MAP


def _load_weights_config() -> dict:
    """Load and cache the weights configuration."""
    return load_yaml_config("weights.yaml")


def _verdict_to_score(verdict: str) -> float:
    """
    Convert a textual verdict to a numeric score in [0, 1].

    Args:
        verdict: One of SUPPORTS, CONTRADICTS, MIXED, etc.

    Returns:
        Numeric score between 0 (definitely false) and 1 (definitely true).
    """
    return _VERDICT_SCORE_MAP.get(verdict.upper(), 0.5)


def _extract_web2_score(web2_evidence: dict) -> float:
    """
    Extract a meaningful score from the potentially nested web2 evidence dict.

    Averages the Google Fact Check and Loki sub-verdicts if both are present.
    """
    details = web2_evidence.get("details", {})
    scores: List[float] = []

    google = details.get("google_fact_check", {})
    loki = details.get("loki_evidence", {})

    for sub in [google, loki]:
        if sub and sub.get("verdict"):
            scores.append(_verdict_to_score(sub["verdict"]))

    if scores:
        return sum(scores) / len(scores)
    # Fallback to top-level verdict
    return _verdict_to_score(web2_evidence.get("verdict", "UNKNOWN"))


def _extract_social_score(social_evidence: dict) -> float:
    """
    Convert social graph evidence to a numeric credibility score.

    Factors:
    - First-poster reputation (higher = more credible)
    - Propagation pattern (normal > low_engagement > astroturfing)
    - Repost count (moderate volume is neutral; extremes are suspicious)
    """
    rep = social_evidence.get("first_poster_reputation", 0.5)
    pattern = social_evidence.get("propagation_pattern", "low_engagement")
    repost_count = social_evidence.get("repost_count", 0)

    # Base score from first-poster reputation
    score = rep

    # Pattern adjustment
    if pattern == "normal":
        score = min(score + 0.1, 1.0)
    elif pattern == "astroturfing":
        score = max(score - 0.25, 0.0)
    # low_engagement: no adjustment

    # Extreme volume adjustment (very high volume with low rep is suspicious)
    if repost_count > 1000 and rep < 0.4:
        score = max(score - 0.1, 0.0)

    return score


def _extract_reputation_score(reputation_evidence: dict) -> float:
    """
    Convert on-chain reputation evidence to a credibility score.

    Uses zScore (0-100) normalized to [0, 1], with aura as a modifier.
    """
    zscore = reputation_evidence.get("zscore", 50)
    aura = reputation_evidence.get("aura", 50)

    # Normalize zScore to [0, 1]
    base = zscore / 100.0

    # Small aura modifier (±0.05)
    aura_modifier = (aura - 50) / 1000.0  # -0.05 to +0.04
    return max(0.0, min(1.0, base + aura_modifier))


# ---------------------------------------------------------------------------
# Dynamic Weight Adjustment
# ---------------------------------------------------------------------------

def dynamic_weight(all_evidences: dict, config: Optional[dict] = None) -> dict:
    """
    Calculate dynamically adjusted weights for each evidence source.

    Adjustments are based on:
    - Quality of each evidence source (whether it returned meaningful data)
    - First-poster reputation (boosts web3_social weight)
    - Propagation pattern abnormalities (reduces web3_social weight)
    - Known source reputation indicators from web2

    Args:
        all_evidences: Dict with keys 'web2', 'web3_social', 'chain_reputation'.
        config: Weights configuration dict (from weights.yaml).

    Returns:
        Dict of adjusted source weights summing to 1.0.
    """
    if config is None:
        config = _load_weights_config()

    defaults = config.get("default_weights", {})
    adjustments = config.get("adjustments", {})

    w_web2 = defaults.get("web2_fact_check", 0.4)
    w_social = defaults.get("web3_social", 0.3)
    w_chain = defaults.get("chain_reputation", 0.3)

    # --- Web3 Social adjustments ---
    social = all_evidences.get("web3_social", {})

    # Boost for high-reputation first poster
    if social.get("first_poster_reputation", 0) >= 0.7:
        boost = adjustments.get("high_reputation_first_poster", 0.15)
        w_social += boost
        logger.debug(f"Applied high_reputation_first_poster boost: +{boost}")

    # Penalty for astroturfing propagation
    if social.get("propagation_pattern") == "astroturfing":
        penalty = adjustments.get("propagation_astroturfing", -0.20)
        w_social += penalty
        logger.debug(f"Applied astroturfing penalty: {penalty}")

    # --- Web2 adjustments ---
    web2 = all_evidences.get("web2", {})
    # If web2 returned no useful evidence, slightly reduce its weight
    if web2.get("verdict") in ("NO_EVIDENCE", "TIMEOUT", "ERROR"):
        w_web2 -= 0.1
        logger.debug("Reduced web2 weight due to missing evidence")

    # --- Chain reputation adjustments ---
    chain = all_evidences.get("chain_reputation", {})
    if chain.get("zscore", 50) >= 80:
        w_chain += 0.05
        logger.debug("Boosted chain_reputation weight for high zScore")

    # --- Normalize weights to sum to 1.0 ---
    total = w_web2 + w_social + w_chain
    if total <= 0:
        logger.warning("All weights zero — falling back to equal weighting")
        return {"web2": 1 / 3, "web3_social": 1 / 3, "chain_reputation": 1 / 3}

    adjusted = {
        "web2": round(w_web2 / total, 4),
        "web3_social": round(w_social / total, 4),
        "chain_reputation": round(w_chain / total, 4),
    }

    logger.info(f"Adjusted weights: {adjusted}")
    return adjusted


# ---------------------------------------------------------------------------
# Conflict Detection
# ---------------------------------------------------------------------------

def resolve_conflicts(evidences: list) -> dict:
    """
    Detect conflicts among evidence sources.

    A conflict exists when two or more sources disagree significantly
    (score difference > max_divergence threshold).

    Args:
        evidences: List of (source_name, score) tuples.

    Returns:
        Dict with 'has_conflict' (bool) and 'conflict_details' (str).
    """
    # Only consider sources with meaningful signals for conflict detection.
    # Neutral scores (0.4-0.6) indicate no strong evidence either way and
    # should not trigger a false conflict.
    _ACTIVE_RANGE = (0.4, 0.6)
    active = [(n, s) for n, s in evidences if s <= _ACTIVE_RANGE[0] or s >= _ACTIVE_RANGE[1]]

    if len(active) < 2:
        return {"has_conflict": False, "conflict_details": "Not enough active sources for conflict check"}

    config = _load_weights_config()
    max_divergence = config.get("conflict", {}).get("max_divergence", 0.4)

    scores = [score for _, score in active]
    min_score = min(scores)
    max_score = max(scores)

    has_conflict = (max_score - min_score) > max_divergence

    if has_conflict:
        sources_high = [name for name, s in active if s == max_score]
        sources_low = [name for name, s in active if s == min_score]
        detail = f"Conflict detected: {sources_high} score ~{max_score:.2f} vs {sources_low} score ~{min_score:.2f}"
        logger.warning(detail)
    else:
        detail = f"Sources are consistent (max divergence {max_score - min_score:.2f} < {max_divergence})"

    return {"has_conflict": has_conflict, "conflict_details": detail}


# ---------------------------------------------------------------------------
# Rating Calculation
# ---------------------------------------------------------------------------

def calculate_rating(scores: dict) -> Tuple[str, float]:
    """
    Calculate the final rating and confidence from weighted evidence scores.

    The final score is a weighted average of individual source scores.
    The rating is determined by threshold comparison:
    - >= 0.75 → TRUST
    - >= 0.25 → SUSPECT
    - < 0.25  → DISTRUST

    Confidence is derived from the distance to the nearest threshold boundary,
    so scores close to boundaries have lower confidence.

    Args:
        scores: Dict mapping source name to {"score": float, "weight": float}.

    Returns:
        Tuple of (rating: str, confidence: float).
    """
    config = _load_weights_config()
    thresholds = config.get("rating_thresholds", {})
    trust_min = thresholds.get("trust_min", 0.75)
    suspect_min = thresholds.get("suspect_min", 0.25)

    # Calculate weighted average
    total_weight = 0.0
    weighted_sum = 0.0
    for source_name, data in scores.items():
        if not isinstance(data, dict):
            continue  # Skip non-dict entries (e.g., conflict_penalty float)
        weight = data.get("weight", 0.0)
        score = data.get("score", 0.5)
        weighted_sum += score * weight
        total_weight += weight

    if total_weight == 0:
        logger.warning("Total weight is zero — returning neutral SUSPECT")
        return ("SUSPECT", 0.5)

    final_score = weighted_sum / total_weight

    # Apply conflict penalty if applicable
    # (checked externally and passed in via scores metadata, or handled in main.py)
    if "conflict_penalty" in scores:
        final_score = max(0.0, final_score + scores["conflict_penalty"])

    # Determine rating
    if final_score >= trust_min:
        rating = "TRUST"
    elif final_score >= suspect_min:
        rating = "SUSPECT"
    else:
        rating = "DISTRUST"

    # Confidence: distance from nearest threshold boundary, scaled to [0, 1]
    # 0.0 = exactly on a boundary; 1.0 = far from any boundary
    distance_to_trust = abs(final_score - trust_min)
    distance_to_suspect = abs(final_score - suspect_min)
    min_distance = min(distance_to_trust, distance_to_suspect)

    # Scale: max meaningful distance is ~0.5 (middle of the range)
    confidence = round(1.0 - min(min_distance / 0.5, 1.0), 3)

    logger.info(f"Final weighted score: {final_score:.4f} → {rating} (confidence: {confidence})")
    return rating, confidence


# ---------------------------------------------------------------------------
# Full scoring pipeline
# ---------------------------------------------------------------------------

def score_evidence(all_evidences: dict) -> dict:
    """
    Run the complete multi-source scoring pipeline.

    1. Convert each source's evidence to a numeric score.
    2. Compute dynamically adjusted weights.
    3. Detect and handle conflicts.
    4. Calculate final rating and confidence.

    Args:
        all_evidences: Evidence dict from evidence_retriever.retrieve_all_evidence().

    Returns:
        Scoring result: {
            "overall_rating": str,
            "confidence": float,
            "source_scores": dict,
            "conflict_info": dict,
        }
    """
    # Step 1: Extract numeric scores from each evidence source
    web2_score = _extract_web2_score(all_evidences.get("web2", {}))
    social_score = _extract_social_score(all_evidences.get("web3_social", {}))
    chain_score = _extract_reputation_score(all_evidences.get("chain_reputation", {}))

    logger.info(f"Raw source scores — web2: {web2_score:.3f}, social: {social_score:.3f}, chain: {chain_score:.3f}")

    # Step 2: Dynamic weight adjustment
    weights = dynamic_weight(all_evidences)

    # Step 3: Check for conflicts
    conflict_info = resolve_conflicts([
        ("web2", web2_score),
        ("web3_social", social_score),
        ("chain_reputation", chain_score),
    ])

    # Step 4: Calculate rating with weights
    scores_input = {
        "web2": {"score": web2_score, "weight": weights["web2"]},
        "web3_social": {"score": social_score, "weight": weights["web3_social"]},
        "chain_reputation": {"score": chain_score, "weight": weights["chain_reputation"]},
    }

    # Apply conflict penalty
    if conflict_info["has_conflict"]:
        config = _load_weights_config()
        penalty = config.get("conflict", {}).get("conflict_penalty", -0.15)
        scores_input["conflict_penalty"] = penalty
        logger.info(f"Applied conflict penalty: {penalty}")

    rating, confidence = calculate_rating(scores_input)

    return {
        "overall_rating": rating,
        "confidence": confidence,
        "source_scores": {
            "web2": round(web2_score, 3),
            "web3_social": round(social_score, 3),
            "chain_reputation": round(chain_score, 3),
        },
        "weights_used": weights,
        "conflict_info": conflict_info,
    }
