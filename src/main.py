"""
Main entry point for the Multi-Source Fact Checker Skill.

Exports the primary function `verify_claim()` that orchestrates the full
fact-checking pipeline: preprocessing → parallel evidence retrieval →
multi-source scoring → grounding validation → final JSON output.

Designed for the Pharos + Anvita Flow Hackathon.
"""

import asyncio
from typing import Any, Dict, List, Optional

from loguru import logger

from .claim_processor import normalize_claim, is_checkable, decompose_claims
from .evidence_retriever import retrieve_all_evidence
from .grounding_validator import check_grounding, build_evidence_text
from .multi_source_scorer import score_evidence
from .utils import utc_iso_timestamp, load_env


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def verify_claim(
    claim_text: str,
    source_info: Optional[dict] = None,
) -> dict:
    """
    Verify a textual claim by cross-referencing multiple independent sources.

    Pipeline:
    1. Normalize and decompose the claim into atomic sub-claims.
    2. Retrieve evidence from three parallel channels (web2, web3 social, chain).
    3. Score and fuse multi-source evidence with dynamic weighting.
    4. Validate grounding consistency between evidence and claim.
    5. Assemble and return the final structured output.

    Args:
        claim_text: The claim to verify, as a plain-text string.
        source_info: Optional context dict. Keys may include:
            - 'addr': Ethereum wallet address for on-chain reputation lookup.
            - 'url' : Original URL where the claim was found (for metadata).

    Returns:
        dict following the multi_source_fact_checker_v1 output schema:
        {
            "skill": "multi_source_fact_checker_v1",
            "timestamp": "ISO-8601",
            "original_claim": str,
            "atomic_claims": [...],
            "overall_rating": "TRUST | SUSPECT | DISTRUST",
            "confidence": float,
            "detailed_breakdown": {...},
            "evidence_trail": [...],
            "partial_data": bool
        }

    Raises:
        ValueError: If claim_text is empty or not a string.
    """
    if not isinstance(claim_text, str):
        raise ValueError(f"claim_text must be a string, got {type(claim_text).__name__}")
    if not claim_text.strip():
        raise ValueError("claim_text must not be empty")

    # Ensure environment is loaded
    load_env()

    partial_data = False

    # ------------------------------------------------------------------
    # Step 1: Normalize the claim
    # ------------------------------------------------------------------
    logger.info("=== Step 1: Claim Normalization ===")
    try:
        normalized = normalize_claim(claim_text)
    except Exception as exc:
        logger.error(f"Normalization failed: {exc}")
        return _error_output(claim_text, f"Normalization error: {exc}")

    # ------------------------------------------------------------------
    # Step 2: Check if the claim is fact-checkable
    # ------------------------------------------------------------------
    logger.info("=== Step 2: Checkability Assessment ===")
    if not is_checkable(normalized):
        logger.info("Claim deemed non-checkable — returning SUSPECT with low confidence")
        return {
            "skill": "multi_source_fact_checker_v1",
            "timestamp": utc_iso_timestamp(),
            "original_claim": claim_text.strip(),
            "atomic_claims": [normalized],
            "overall_rating": "SUSPECT",
            "confidence": 0.25,
            "detailed_breakdown": {
                "supported_by": [],
                "contradicted_by": [],
                "grounding_score": 0.5,
            },
            "evidence_trail": [
                {
                    "source": "system",
                    "verdict": "NOT_CHECKABLE",
                    "details": {"reason": "Claim is subjective, normative, or a future prediction"},
                }
            ],
            "partial_data": False,
        }

    # ------------------------------------------------------------------
    # Step 3: Decompose into atomic claims
    # ------------------------------------------------------------------
    logger.info("=== Step 3: Claim Decomposition ===")
    try:
        atomic_claims = decompose_claims(normalized)
    except Exception as exc:
        logger.error(f"Decomposition failed: {exc}")
        atomic_claims = [normalized]
        partial_data = True

    # ------------------------------------------------------------------
    # Step 4: Parallel evidence retrieval (async → sync bridge)
    # ------------------------------------------------------------------
    logger.info("=== Step 4: Evidence Retrieval ===")
    try:
        all_evidences = asyncio.run(
            retrieve_all_evidence(normalized, source_info=source_info, total_timeout=8.0)
        )
    except Exception as exc:
        logger.error(f"Evidence retrieval error: {exc}")
        partial_data = True
        all_evidences = _empty_evidence()

    # Check for timeouts / partial results
    for source_key, ev in all_evidences.items():
        if ev.get("verdict") in ("TIMEOUT", "ERROR"):
            partial_data = True
            logger.warning(f"Source '{source_key}' returned {ev.get('verdict')}")

    # ------------------------------------------------------------------
    # Step 5: Multi-source scoring
    # ------------------------------------------------------------------
    logger.info("=== Step 5: Multi-Source Scoring ===")
    try:
        scoring_result = score_evidence(all_evidences)
    except Exception as exc:
        logger.error(f"Scoring failed: {exc}")
        partial_data = True
        scoring_result = {
            "overall_rating": "SUSPECT",
            "confidence": 0.3,
            "source_scores": {},
            "weights_used": {},
            "conflict_info": {"has_conflict": False, "conflict_details": str(exc)},
        }

    # ------------------------------------------------------------------
    # Step 6: Grounding validation
    # ------------------------------------------------------------------
    logger.info("=== Step 6: Grounding Validation ===")
    try:
        evidence_text = build_evidence_text(all_evidences)
        grounding_score = check_grounding(normalized, evidence_text)
    except Exception as exc:
        logger.error(f"Grounding validation failed: {exc}")
        grounding_score = 0.5
        partial_data = True

    # ------------------------------------------------------------------
    # Step 7: Build evidence trail
    # ------------------------------------------------------------------
    evidence_trail = _build_evidence_trail(all_evidences)

    # Build supported_by / contradicted_by lists
    supported_by: List[str] = []
    contradicted_by: List[str] = []
    for entry in evidence_trail:
        verdict = entry.get("verdict", "").upper()
        if verdict in ("SUPPORTS", "LIKELY_SUPPORTS"):
            supported_by.append(entry["source"])
        elif verdict in ("CONTRADICTS", "LIKELY_CONTRADICTS"):
            contradicted_by.append(entry["source"])

    # ------------------------------------------------------------------
    # Step 8: Assemble final output
    # ------------------------------------------------------------------
    output = {
        "skill": "multi_source_fact_checker_v1",
        "timestamp": utc_iso_timestamp(),
        "original_claim": claim_text.strip(),
        "atomic_claims": atomic_claims,
        "overall_rating": scoring_result["overall_rating"],
        "confidence": scoring_result["confidence"],
        "detailed_breakdown": {
            "supported_by": supported_by,
            "contradicted_by": contradicted_by,
            "grounding_score": grounding_score,
        },
        "evidence_trail": evidence_trail,
        "partial_data": partial_data,
    }

    logger.info(
        f"=== Verification Complete: {output['overall_rating']} "
        f"(confidence: {output['confidence']}, partial_data: {partial_data}) ==="
    )
    return output


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_evidence_trail(all_evidences: dict) -> list:
    """
    Build the evidence_trail list from the raw evidence dict.

    Each entry describes one source and its contribution.
    """
    trail: list = []

    # Web2 trail
    web2 = all_evidences.get("web2", {})
    trail.append({
        "source": "web2",
        "verdict": web2.get("verdict", "UNKNOWN"),
        "details": web2.get("details", {}),
    })

    # Web3 Social trail
    social = all_evidences.get("web3_social", {})
    trail.append({
        "source": "web3_social",
        "verdict": _social_to_verdict(social),
        "details": {
            "repost_count": social.get("repost_count", 0),
            "first_poster_reputation": social.get("first_poster_reputation", 0.5),
            "propagation_pattern": social.get("propagation_pattern", "unknown"),
        },
    })

    # Chain reputation trail
    chain = all_evidences.get("chain_reputation", {})
    trail.append({
        "source": "chain_reputation",
        "verdict": _reputation_to_verdict(chain),
        "details": {
            "zscore": chain.get("zscore", 50),
            "aura": chain.get("aura", 50),
            "payment_backing": chain.get("payment_backing", 0.0),
            "address": chain.get("details", {}).get("address", "N/A"),
        },
    })

    return trail


def _social_to_verdict(social_evidence: dict) -> str:
    """Convert social graph evidence to a summary verdict."""
    rep = social_evidence.get("first_poster_reputation", 0.5)
    pattern = social_evidence.get("propagation_pattern", "low_engagement")

    if pattern == "astroturfing":
        return "LIKELY_CONTRADICTS"
    elif rep >= 0.7 and pattern == "normal":
        return "LIKELY_SUPPORTS"
    elif rep <= 0.3:
        return "LIKELY_CONTRADICTS"
    else:
        return "MIXED"


def _reputation_to_verdict(reputation_evidence: dict) -> str:
    """Convert on-chain reputation evidence to a summary verdict."""
    zscore = reputation_evidence.get("zscore", 50)
    if zscore >= 70:
        return "LIKELY_SUPPORTS"
    elif zscore <= 30:
        return "LIKELY_CONTRADICTS"
    else:
        return "MIXED"


def _empty_evidence() -> dict:
    """Return a default evidence dict for when retrieval fails entirely."""
    return {
        "web2": {"source": "web2", "verdict": "ERROR", "details": {"error": "retrieval failed"}},
        "web3_social": {"source": "web3_social", "verdict": "ERROR", "details": {"error": "retrieval failed"}},
        "chain_reputation": {"source": "chain_reputation", "verdict": "ERROR", "details": {"error": "retrieval failed"}},
    }


def _error_output(claim_text: str, error_msg: str) -> dict:
    """Return a minimal error output dict."""
    return {
        "skill": "multi_source_fact_checker_v1",
        "timestamp": utc_iso_timestamp(),
        "original_claim": claim_text,
        "atomic_claims": [],
        "overall_rating": "SUSPECT",
        "confidence": 0.0,
        "detailed_breakdown": {
            "supported_by": [],
            "contradicted_by": [],
            "grounding_score": 0.0,
        },
        "evidence_trail": [
            {"source": "system", "verdict": "ERROR", "details": {"error": error_msg}},
        ],
        "partial_data": True,
    }


# ---------------------------------------------------------------------------
# Convenience: sync launcher for CLI usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        claim = " ".join(sys.argv[1:])
    else:
        claim = input("Enter a claim to verify: ")

    result = verify_claim(claim)
    import json

    print(json.dumps(result, indent=2, ensure_ascii=False))
