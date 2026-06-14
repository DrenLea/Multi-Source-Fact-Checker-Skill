"""
Unit tests for the Multi-Source Fact Checker Skill.

Covers the full pipeline and edge cases. Run with:

    pytest tests/ -v
"""

import sys
import os

# Ensure the project root is on the path so "from src.xxx import ..." works
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from src.claim_processor import normalize_claim, is_checkable, decompose_claims
from src.multi_source_scorer import (
    _verdict_to_score,
    _extract_web2_score,
    _extract_social_score,
    _extract_reputation_score,
    resolve_conflicts,
    calculate_rating,
    dynamic_weight,
)
from src.grounding_validator import check_grounding, build_evidence_text
from src.main import verify_claim


# -------------------------------------------------------------------
# Test Data
# -------------------------------------------------------------------

SAMPLE_TRUST_CLAIM = "Bitcoin is a cryptocurrency launched in 2009."
SAMPLE_FALSE_CLAIM = "Ethereum will crash to zero on January 1st 2025 and become completely worthless."
SAMPLE_SUSPECT_CLAIM = "The project may do an airdrop in June 2026 but nothing is officially confirmed."
SAMPLE_SUBJECTIVE = "I think Bitcoin is the best investment ever."
SAMPLE_FUTURE = "Bitcoin will skyrocket to one million dollars by 2030."
SAMPLE_EMPTY = ""
SAMPLE_LONG = "Ethereum is a decentralized blockchain platform. " * 100


# -------------------------------------------------------------------
# 1. Claim Processor Tests
# -------------------------------------------------------------------

class TestClaimProcessor:
    """Tests for claim_processor.py"""

    def test_normalize_strips_whitespace(self):
        result = normalize_claim("   Bitcoin   is  digital   gold   ")
        assert result == 'Bitcoin is digital gold'

    def test_normalize_collapses_newlines(self):
        result = normalize_claim("Line one.\n\nLine two.\nLine three.")
        assert result == "Line one. Line two. Line three."

    def test_normalize_curly_quotes(self):
        result = normalize_claim('“Hello world”')
        assert result == '"Hello world"'

    def test_normalize_rejects_non_string(self):
        with pytest.raises(TypeError):
            normalize_claim(12345)  # type: ignore

    def test_is_checkable_true_for_factual(self):
        assert is_checkable("Bitcoin is a cryptocurrency launched in 2009.") is True

    def test_is_checkable_false_for_subjective(self):
        assert is_checkable("I think Bitcoin is the best investment ever.") is False

    def test_is_checkable_false_for_question(self):
        assert is_checkable("Will Ethereum go up tomorrow?") is False

    def test_is_checkable_false_for_short(self):
        assert is_checkable("Hi") is False

    def test_is_checkable_false_for_future_prediction(self):
        assert is_checkable("Bitcoin will skyrocket to one million by 2030.") is False

    def test_decompose_simple(self):
        result = decompose_claims("Bitcoin is a cryptocurrency.")
        assert len(result) >= 1
        assert "Bitcoin is a cryptocurrency" in result[0]

    def test_decompose_compound(self):
        result = decompose_claims(
            "Bitcoin is a cryptocurrency. Ethereum supports smart contracts."
        )
        assert len(result) >= 2


# -------------------------------------------------------------------
# 2. Scorer Tests
# -------------------------------------------------------------------

class TestScorer:
    """Tests for multi_source_scorer.py"""

    def test_verdict_to_score_supports(self):
        assert _verdict_to_score("SUPPORTS") == 1.0

    def test_verdict_to_score_contradicts(self):
        assert _verdict_to_score("CONTRADICTS") == 0.0

    def test_verdict_to_score_mixed(self):
        assert _verdict_to_score("MIXED") == 0.5

    def test_verdict_to_score_unknown(self):
        assert _verdict_to_score("UNKNOWN") == 0.5

    def test_extract_web2_score_with_data(self):
        evidence = {
            "source": "web2",
            "verdict": "SUPPORTS",
            "details": {
                "google_fact_check": {
                    "source": "google_fact_check",
                    "verdict": "SUPPORTS",
                    "details": {"publisher": "Reuters", "textual_rating": "true"},
                },
                "loki_evidence": {
                    "source": "loki",
                    "verdict": "SUPPORTS",
                    "details": {},
                },
            },
        }
        score = _extract_web2_score(evidence)
        assert score == 1.0

    def test_extract_social_score_high_rep(self):
        evidence = {
            "source": "web3_social",
            "first_poster_reputation": 0.85,
            "propagation_pattern": "normal",
            "repost_count": 500,
        }
        score = _extract_social_score(evidence)
        assert score >= 0.85  # High rep + normal pattern

    def test_extract_social_score_astroturfing(self):
        evidence = {
            "source": "web3_social",
            "first_poster_reputation": 0.2,
            "propagation_pattern": "astroturfing",
            "repost_count": 2000,
        }
        score = _extract_social_score(evidence)
        assert score <= 0.3  # Low rep + astroturfing

    def test_extract_reputation_score_high(self):
        evidence = {"source": "chain_reputation", "zscore": 85, "aura": 70}
        score = _extract_reputation_score(evidence)
        assert score >= 0.8

    def test_extract_reputation_score_low(self):
        evidence = {"source": "chain_reputation", "zscore": 20, "aura": 30}
        score = _extract_reputation_score(evidence)
        assert score <= 0.3

    def test_resolve_conflicts_no_conflict(self):
        evidences = [("web2", 0.8), ("social", 0.85), ("chain", 0.9)]
        result = resolve_conflicts(evidences)
        assert result["has_conflict"] is False

    def test_resolve_conflicts_yes_conflict(self):
        evidences = [("web2", 0.9), ("social", 0.1), ("chain", 0.85)]
        result = resolve_conflicts(evidences)
        assert result["has_conflict"] is True

    def test_calculate_rating_trust(self):
        scores = {
            "web2": {"score": 0.9, "weight": 0.4},
            "web3_social": {"score": 0.85, "weight": 0.3},
            "chain_reputation": {"score": 0.9, "weight": 0.3},
        }
        rating, confidence = calculate_rating(scores)
        assert rating == "TRUST"
        assert 0.0 <= confidence <= 1.0

    def test_calculate_rating_distrust(self):
        scores = {
            "web2": {"score": 0.1, "weight": 0.4},
            "web3_social": {"score": 0.05, "weight": 0.3},
            "chain_reputation": {"score": 0.1, "weight": 0.3},
        }
        rating, confidence = calculate_rating(scores)
        assert rating == "DISTRUST"

    def test_dynamic_weight_sums_to_one(self):
        all_evidences = {
            "web2": {"verdict": "SUPPORTS", "details": {}},
            "web3_social": {"first_poster_reputation": 0.85, "propagation_pattern": "normal", "repost_count": 500},
            "chain_reputation": {"zscore": 80, "aura": 70},
        }
        weights = dynamic_weight(all_evidences)
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.01


# -------------------------------------------------------------------
# 3. Grounding Validator Tests
# -------------------------------------------------------------------

class TestGroundingValidator:
    """Tests for grounding_validator.py"""

    def test_check_grounding_keywords_perfect_overlap(self):
        claim = "Bitcoin is a cryptocurrency"
        evidence = "Bitcoin is a cryptocurrency that uses blockchain technology."
        score = check_grounding(claim, evidence)
        assert 0.0 <= score <= 1.0

    def test_check_grounding_empty_evidence(self):
        score = check_grounding("Bitcoin is a cryptocurrency", "")
        assert score == 0.5  # Neutral fallback

    def test_build_evidence_text(self):
        all_evidences = {
            "web2": {
                "verdict": "SUPPORTS",
                "details": {
                    "google_fact_check": {
                        "source": "gfc",
                        "verdict": "SUPPORTS",
                        "details": {"publisher": "Reuters", "textual_rating": "true"},
                    },
                    "loki_evidence": {
                        "source": "loki",
                        "verdict": "SUPPORTS",
                        "details": {"top_passage": "Bitcoin is widely used."},
                    },
                },
            },
            "web3_social": {
                "first_poster_reputation": 0.85,
                "propagation_pattern": "normal",
                "repost_count": 500,
            },
            "chain_reputation": {
                "zscore": 80,
                "aura": 70,
                "payment_backing": 5.0,
            },
        }
        text = build_evidence_text(all_evidences)
        assert "Reuters" in text or "true" in text
        assert "Bitcoin is widely used" in text


# -------------------------------------------------------------------
# 4. Integration / Main Pipeline Tests
# -------------------------------------------------------------------

class TestVerifyClaim:
    """End-to-end integration tests for verify_claim()"""

    def test_trustworthy_claim(self):
        """A well-known factual claim should return TRUST."""
        result = verify_claim(SAMPLE_TRUST_CLAIM)
        assert result["skill"] == "multi_source_fact_checker_v1"
        assert "timestamp" in result
        assert result["original_claim"] == SAMPLE_TRUST_CLAIM
        assert len(result["atomic_claims"]) >= 1
        assert result["overall_rating"] == "TRUST"
        assert result["confidence"] > 0.5
        assert len(result["evidence_trail"]) == 3
        assert result["partial_data"] is False

    def test_false_claim(self):
        """A demonstrably false claim should return DISTRUST."""
        result = verify_claim(SAMPLE_FALSE_CLAIM)
        assert result["overall_rating"] in ("DISTRUST", "SUSPECT")
        assert "confidence" in result

    def test_suspect_claim(self):
        """An unverifiable/rumor claim should return SUSPECT."""
        result = verify_claim(SAMPLE_SUSPECT_CLAIM)
        assert result["overall_rating"] == "SUSPECT"
        assert len(result["atomic_claims"]) >= 1

    def test_subjective_claim_not_checkable(self):
        """A subjective opinion should be flagged as not checkable."""
        result = verify_claim(SAMPLE_SUBJECTIVE)
        assert result["overall_rating"] == "SUSPECT"
        # Should have a NOT_CHECKABLE verdict in evidence trail
        verdicts = [e["verdict"] for e in result["evidence_trail"]]
        assert "NOT_CHECKABLE" in verdicts

    def test_empty_string_raises(self):
        """Empty input should raise ValueError."""
        with pytest.raises(ValueError):
            verify_claim("")

    def test_output_schema_compliance(self):
        """Verify the output matches the documented JSON schema."""
        result = verify_claim("Bitcoin is a decentralized digital currency.")
        # Required top-level keys
        for key in [
            "skill", "timestamp", "original_claim", "atomic_claims",
            "overall_rating", "confidence", "detailed_breakdown",
            "evidence_trail", "partial_data",
        ]:
            assert key in result, f"Missing key: {key}"
        # detailed_breakdown sub-keys
        breakdown = result["detailed_breakdown"]
        for key in ["supported_by", "contradicted_by", "grounding_score"]:
            assert key in breakdown, f"Missing breakdown key: {key}"
        # evidence_trail entries
        for entry in result["evidence_trail"]:
            for key in ["source", "verdict", "details"]:
                assert key in entry, f"Missing trail key: {key} in {entry}"
        # Type checks
        assert isinstance(result["confidence"], (int, float))
        assert 0.0 <= result["confidence"] <= 1.0
        assert result["overall_rating"] in ("TRUST", "SUSPECT", "DISTRUST")
        assert isinstance(result["partial_data"], bool)

    def test_with_source_info(self):
        """verify_claim should accept and use source_info with an address."""
        result = verify_claim(
            "This project will distribute tokens soon.",
            source_info={"addr": "0x1234567890abcdef1234567890abcdef67890123", "url": "https://example.com/claim"},
        )
        assert result["skill"] == "multi_source_fact_checker_v1"
        assert result["overall_rating"] in ("TRUST", "SUSPECT", "DISTRUST")

    def test_future_prediction_not_checkable(self):
        """A far-future prediction should be flagged as not checkable."""
        result = verify_claim(SAMPLE_FUTURE)
        verdicts = [e["verdict"] for e in result["evidence_trail"]]
        assert "NOT_CHECKABLE" in verdicts
