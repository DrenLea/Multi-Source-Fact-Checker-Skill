"""
Evidence Retriever Module

Fetches evidence from three parallel channels:
1. Web2 sources — Loki framework + Google Fact Check Tools API
2. Web3 social graph — Lens Protocol + Farcaster
3. On-chain reputation — zScore + TraceRank

All retrievers run concurrently via asyncio.gather with a hard timeout.
Mock implementations are provided for APIs that require authentication;
swap them out by setting mock.enabled=false in api_config.yaml.
"""

import asyncio
import random
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import yaml
from loguru import logger
from pathlib import Path

from .utils import load_yaml_config, resolve_env_vars


# ---------------------------------------------------------------------------
# Load configuration on module load
# ---------------------------------------------------------------------------

_config: dict = {}
_weights_config: dict = {}


def _init_config() -> dict:
    """Lazy-load and cache the API configuration."""
    global _config
    if not _config:
        raw = load_yaml_config("api_config.yaml")
        _config = resolve_env_vars(raw)
    return _config


def _is_mock_enabled() -> bool:
    """Check whether mock mode is enabled in config."""
    cfg = _init_config()
    return cfg.get("mock", {}).get("enabled", True)


# ---------------------------------------------------------------------------
# Helper: enforce timeout on a coroutine
# ---------------------------------------------------------------------------

async def _with_timeout(coro, timeout_sec: float = 5.0, label: str = "task"):
    """
    Run a coroutine with a hard timeout.

    Args:
        coro: Awaitable object.
        timeout_sec: Maximum seconds to wait.
        label: Human-readable label for log messages.

    Returns:
        Result of coro, or a fallback dict on timeout/error.
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout_sec)
    except asyncio.TimeoutError:
        logger.warning(f"[{label}] Timed out after {timeout_sec}s")
        return {"source": label, "verdict": "TIMEOUT", "details": {"error": "timeout"}}
    except Exception as exc:
        logger.error(f"[{label}] Error: {exc}")
        return {"source": label, "verdict": "ERROR", "details": {"error": str(exc)}}


# ---------------------------------------------------------------------------
# 1. Web2 Retrieval
# ---------------------------------------------------------------------------

async def retrieve_from_web2(claim: str) -> dict:
    """
    Retrieve evidence from traditional web2 fact-checking sources.

    Channels:
    - Google Fact Check Tools API (primary)
    - Loki framework evidence retrieval (fallback / supplementary)

    Args:
        claim: Atomic claim string.

    Returns:
        Evidence dict: {"source": "web2", "verdict": str, "details": {...}}
    """
    logger.info(f"[web2] Retrieving evidence for: {claim[:60]}...")

    if _is_mock_enabled():
        return await _mock_web2_retrieval(claim)

    # --- Real implementation (requires API keys) ---
    google_result = await _with_timeout(
        _google_fact_check(claim), timeout_sec=5.0, label="web2_google"
    )
    loki_result = await _with_timeout(
        _loki_retrieve(claim), timeout_sec=5.0, label="web2_loki"
    )

    # Merge results — prefer Google's verdict if available
    merged = _merge_web2_results(google_result, loki_result)
    return merged


async def _google_fact_check(claim: str) -> dict:
    """
    Call the Google Fact Check Tools API.

    API docs: https://developers.google.com/fact-check/tools/api/reference/rest
    """
    cfg = _init_config()
    api_key = cfg.get("google_fact_check", {}).get("api_key", "")
    base_url = cfg.get("google_fact_check", {}).get("base_url", "")

    if not api_key or "your_" in api_key:
        logger.warning("[web2] Google Fact Check API key not configured — using fallback")
        return {"source": "google_fact_check", "verdict": "UNKNOWN", "details": {}}

    import aiohttp

    params = {"query": claim, "key": api_key, "languageCode": "en"}
    async with aiohttp.ClientSession() as session:
        async with session.get(base_url, params=params) as resp:
            if resp.status == 200:
                data = await resp.json()
                return _parse_google_fact_check_response(data)
            else:
                logger.warning(f"[web2] Google API returned {resp.status}")
                return {"source": "google_fact_check", "verdict": "UNKNOWN", "details": {}}


def _parse_google_fact_check_response(data: dict) -> dict:
    """Parse the Google Fact Check Tools API response into our evidence format."""
    claims = data.get("claims", [])
    if not claims:
        return {"source": "google_fact_check", "verdict": "NO_EVIDENCE", "details": {}}

    # Take the first claim review
    review = claims[0].get("claimReview", [{}])[0]
    textual_rating = review.get("textualRating", "").lower()
    publisher = review.get("publisher", {}).get("name", "unknown")
    url = review.get("url", "")

    # Map common Google ratings to our simplified verdict
    if any(w in textual_rating for w in ["true", "correct", "accurate", "confirmed", "fact"]):
        verdict = "SUPPORTS"
    elif any(w in textual_rating for w in ["false", "fake", "incorrect", "inaccurate", "false"]):
        verdict = "CONTRADICTS"
    elif any(w in textual_rating for w in ["partly", "mixed", "half", "misleading", "missing context"]):
        verdict = "MIXED"
    else:
        verdict = "UNKNOWN"

    return {
        "source": "google_fact_check",
        "verdict": verdict,
        "details": {
            "publisher": publisher,
            "url": url,
            "textual_rating": textual_rating,
        },
    }


async def _loki_retrieve(claim: str) -> dict:
    """
    Retrieve evidence using the Loki framework.

    If Loki is not installed or the model is unavailable, returns NO_EVIDENCE.
    Replace this mock with the actual loki.retrieve_evidence() call.
    """
    try:
        # Real call (uncomment when loki is installed):
        # from loki import EvidenceRetriever
        # retriever = EvidenceRetriever()
        # result = retriever.retrieve(claim, top_k=3)
        # return _format_loki_result(result)
        raise ImportError("Loki not installed")
    except (ImportError, Exception) as exc:
        logger.debug(f"[web2] Loki unavailable: {exc}")
        return {"source": "loki", "verdict": "NO_EVIDENCE", "details": {"reason": str(exc)}}


def _merge_web2_results(google_result: dict, loki_result: dict) -> dict:
    """Merge results from multiple web2 sources into a single evidence dict."""
    verdicts = [google_result.get("verdict"), loki_result.get("verdict")]
    # Prefer Google's verdict if it has a meaningful one
    primary = google_result if google_result.get("verdict") not in ("NO_EVIDENCE", "UNKNOWN") else loki_result

    return {
        "source": "web2",
        "verdict": primary.get("verdict", "UNKNOWN"),
        "details": {
            "google_fact_check": google_result,
            "loki_evidence": loki_result,
        },
    }


# ---------------------------------------------------------------------------
# 2. Web3 Social Graph Retrieval
# ---------------------------------------------------------------------------

async def retrieve_from_social_graph(claim: str) -> dict:
    """
    Retrieve evidence from Web3 social protocols (Lens Protocol, Farcaster).

    Simulates checking:
    - How widely the claim has been reposted
    - Reputation of the first poster (via OpenRank / on-chain rep)
    - Propagation pattern (organic vs. coordinated/astroturfing)

    Args:
        claim: Atomic claim string.

    Returns:
        Evidence dict with propagation metrics.
    """
    logger.info(f"[web3_social] Analyzing social graph for: {claim[:60]}...")

    if _is_mock_enabled():
        return await _mock_social_graph_retrieval(claim)

    lens_task = _with_timeout(_query_lens(claim), timeout_sec=5.0, label="web3_lens")
    farcaster_task = _with_timeout(_query_farcaster(claim), timeout_sec=5.0, label="web3_farcaster")

    lens_result, fc_result = await asyncio.gather(lens_task, farcaster_task)
    return _merge_social_results(lens_result, fc_result)


async def _query_lens(claim: str) -> dict:
    """Query Lens Protocol for posts matching the claim."""
    cfg = _init_config()
    api_key = cfg.get("lens_protocol", {}).get("api_key", "")

    if not api_key or "your_" in api_key:
        logger.debug("[web3_social] Lens API key not configured")
        return {"source": "lens", "post_count": 0, "first_poster_reputation": 0.0}

    # Real Lens API query would go here
    # import aiohttp
    # async with aiohttp.ClientSession() as session:
    #     ...
    return {"source": "lens", "post_count": 0, "first_poster_reputation": 0.0}


async def _query_farcaster(claim: str) -> dict:
    """Query Farcaster Hub for casts matching the claim."""
    cfg = _init_config()
    hub_url = cfg.get("farcaster", {}).get("hub_url", "")

    if not hub_url:
        logger.debug("[web3_social] Farcaster Hub URL not configured")
        return {"source": "farcaster", "cast_count": 0, "first_poster_reputation": 0.0}

    # Real Farcaster query would go here
    return {"source": "farcaster", "cast_count": 0, "first_poster_reputation": 0.0}


def _merge_social_results(lens_result: dict, farcaster_result: dict) -> dict:
    """Merge Lens and Farcaster results into a unified social graph evidence dict."""
    total_posts = lens_result.get("post_count", 0) + farcaster_result.get("cast_count", 0)

    # Average first-poster reputation
    reps = [
        lens_result.get("first_poster_reputation", 0.0),
        farcaster_result.get("first_poster_reputation", 0.0),
    ]
    avg_rep = sum(reps) / len([r for r in reps if r > 0]) if any(r > 0 for r in reps) else 0.5

    # Heuristic: low rep + high volume → possible astroturfing
    if total_posts > 100 and avg_rep < 0.3:
        pattern = "astroturfing"
    elif total_posts > 10:
        pattern = "normal"
    else:
        pattern = "low_engagement"

    return {
        "source": "web3_social",
        "repost_count": total_posts,
        "first_poster_reputation": round(avg_rep, 3),
        "propagation_pattern": pattern,
        "details": {
            "lens": lens_result,
            "farcaster": farcaster_result,
        },
    }


# ---------------------------------------------------------------------------
# 3. On-Chain Reputation Retrieval
# ---------------------------------------------------------------------------

async def retrieve_from_chain_reputation(addr: Optional[str] = None) -> dict:
    """
    Retrieve on-chain reputation metrics for a given wallet address.

    Uses:
    - zScore: Comprehensive on-chain reputation score (0-100)
    - TraceRank: Transaction graph ranking

    If no address is provided, returns a neutral default.

    Args:
        addr: Ethereum wallet address (0x...).

    Returns:
        Evidence dict: {"source": "chain_reputation", "zscore": int, "aura": int, ...}
    """
    if not addr:
        logger.info("[chain_reputation] No address provided — returning neutral score")
        return {
            "source": "chain_reputation",
            "zscore": 50,
            "aura": 50,
            "payment_backing": 0.0,
            "details": {"note": "no_address_provided"},
        }

    logger.info(f"[chain_reputation] Fetching on-chain data for {addr[:10]}...")

    if _is_mock_enabled():
        return await _mock_chain_reputation_retrieval(addr)

    zscore_task = _with_timeout(_query_zscore(addr), timeout_sec=5.0, label="chain_zscore")
    tracerank_task = _with_timeout(_query_tracerank(addr), timeout_sec=5.0, label="chain_tracerank")

    zscore_result, tracerank_result = await asyncio.gather(zscore_task, tracerank_task)
    return _merge_reputation_results(zscore_result, tracerank_result, addr)


async def _query_zscore(addr: str) -> dict:
    """Query zScore API for on-chain reputation."""
    cfg = _init_config()
    rpc_url = cfg.get("zscore", {}).get("rpc_url", "")

    if not rpc_url:
        logger.debug("[chain_reputation] zScore RPC URL not configured")
        return {"zscore": 50, "aura": 50, "payment_backing": 0.0}

    # Real zScore API call:
    # import aiohttp
    # async with aiohttp.ClientSession() as session:
    #     async with session.get(f"{rpc_url}/score/{addr}") as resp:
    #         ...
    return {"zscore": 50, "aura": 50, "payment_backing": 0.0}


async def _query_tracerank(addr: str) -> dict:
    """Query TraceRank for transaction graph reputation."""
    cfg = _init_config()
    endpoint = cfg.get("tracerank", {}).get("endpoint", "")

    if not endpoint:
        logger.debug("[chain_reputation] TraceRank endpoint not configured")
        return {"rank": 0.5, "volume": 0.0}

    # Real TraceRank API call would go here
    return {"rank": 0.5, "volume": 0.0}


def _merge_reputation_results(zscore: dict, tracerank: dict, addr: str) -> dict:
    """Merge zScore and TraceRank into a unified reputation evidence dict."""
    return {
        "source": "chain_reputation",
        "zscore": zscore.get("zscore", 50),
        "aura": zscore.get("aura", 50),
        "payment_backing": zscore.get("payment_backing", 0.0),
        "details": {
            "address": addr,
            "zscore_raw": zscore,
            "tracerank": tracerank,
        },
    }


# ---------------------------------------------------------------------------
# Orchestrator: run all three retrievers in parallel
# ---------------------------------------------------------------------------

async def retrieve_all_evidence(
    claim: str,
    source_info: Optional[dict] = None,
    total_timeout: float = 8.0,
) -> dict:
    """
    Execute all three evidence retrievers concurrently.

    Args:
        claim: Atomic claim string.
        source_info: Optional dict containing extra context (e.g. {'addr': '0x...'}).
        total_timeout: Maximum total time in seconds for all retrievals.

    Returns:
        Dict mapping source names to their evidence results:
        {"web2": {...}, "web3_social": {...}, "chain_reputation": {...}}
    """
    addr = source_info.get("addr") if source_info else None

    logger.info(f"Starting parallel evidence retrieval (timeout={total_timeout}s)...")

    tasks = [
        _with_timeout(retrieve_from_web2(claim), timeout_sec=5.0, label="web2"),
        _with_timeout(retrieve_from_social_graph(claim), timeout_sec=5.0, label="web3_social"),
        _with_timeout(retrieve_from_chain_reputation(addr), timeout_sec=5.0, label="chain_reputation"),
    ]

    try:
        results = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=False),
            timeout=total_timeout,
        )
    except asyncio.TimeoutError:
        logger.warning(f"Total evidence retrieval exceeded {total_timeout}s — using partial results")
        results = [
            {"source": "web2", "verdict": "TIMEOUT", "details": {"error": "total_timeout"}},
            {"source": "web3_social", "verdict": "TIMEOUT", "details": {"error": "total_timeout"}},
            {"source": "chain_reputation", "verdict": "TIMEOUT", "details": {"error": "total_timeout"}},
        ]

    evidence = {
        "web2": results[0],
        "web3_social": results[1],
        "chain_reputation": results[2],
    }

    logger.info("Evidence retrieval complete")
    return evidence


# ---------------------------------------------------------------------------
# Mock Implementations
# ---------------------------------------------------------------------------

async def _mock_web2_retrieval(claim: str) -> dict:
    """
    Mock web2 evidence retrieval with realistic sample data.

    Replace this with real API calls by setting mock.enabled=false in config.
    """
    # Simulate network latency
    await asyncio.sleep(random.uniform(0.1, 0.5))

    # Keyword-based mock responses for demonstration purposes
    claim_lower = claim.lower()

    # Simple heuristic matching for demo
    if any(w in claim_lower for w in ["bitcoin", "btc", "cryptocurrency"]):
        if any(w in claim_lower for w in ["cryptocurrenc", "digital", "currency", "asset"]):
            return {
                "source": "web2",
                "verdict": "SUPPORTS",
                "details": {
                    "google_fact_check": {
                        "source": "google_fact_check",
                        "verdict": "SUPPORTS",
                        "details": {"publisher": "CoinDesk", "url": "https://www.coindesk.com/", "textual_rating": "true"},
                    },
                    "loki_evidence": {
                        "source": "loki",
                        "verdict": "SUPPORTS",
                        "details": {"evidence_count": 5, "top_passage": "Bitcoin is widely recognized as a cryptocurrency and digital asset."},
                    },
                },
            }

    if any(w in claim_lower for w in ["zero", "crash", "collapse", "die", "worthless"]):
        return {
            "source": "web2",
            "verdict": "CONTRADICTS",
            "details": {
                "google_fact_check": {
                    "source": "google_fact_check",
                    "verdict": "CONTRADICTS",
                    "details": {"publisher": "Reuters Fact Check", "url": "https://www.reuters.com/fact-check/", "textual_rating": "false"},
                },
                "loki_evidence": {
                    "source": "loki",
                    "verdict": "CONTRADICTS",
                    "details": {"evidence_count": 3, "top_passage": "No evidence supports the claim that the asset will become worthless."},
                },
            },
        }

    if any(w in claim_lower for w in ["airdrop", "may", "might", "could", "rumor", "potential"]):
        return {
            "source": "web2",
            "verdict": "MIXED",
            "details": {
                "google_fact_check": {
                    "source": "google_fact_check",
                    "verdict": "MIXED",
                    "details": {"publisher": "Snopes", "url": "https://www.snopes.com/", "textual_rating": "unverified"},
                },
                "loki_evidence": {
                    "source": "loki",
                    "verdict": "MIXED",
                    "details": {"evidence_count": 2, "top_passage": "The project has not officially confirmed any airdrop plans."},
                },
            },
        }

    # Default: no strong evidence either way
    return {
        "source": "web2",
        "verdict": "NO_EVIDENCE",
        "details": {
            "google_fact_check": {"source": "google_fact_check", "verdict": "NO_EVIDENCE", "details": {}},
            "loki_evidence": {"source": "loki", "verdict": "NO_EVIDENCE", "details": {}},
        },
    }


async def _mock_social_graph_retrieval(claim: str) -> dict:
    """
    Mock Web3 social graph analysis.

    Returns plausible simulated data based on claim content.
    """
    await asyncio.sleep(random.uniform(0.1, 0.3))
    claim_lower = claim.lower()

    # High-engagement for well-known topics
    if any(w in claim_lower for w in ["bitcoin", "ethereum", "eth", "btc"]):
        repost_count = random.randint(500, 2000)
        first_poster_reputation = round(random.uniform(0.6, 0.95), 3)
        pattern = "normal"
    elif any(w in claim_lower for w in ["zero", "crash", "collapse", "die", "rug", "scam"]):
        repost_count = random.randint(50, 300)
        first_poster_reputation = round(random.uniform(0.1, 0.35), 3)
        pattern = "astroturfing"
    elif any(w in claim_lower for w in ["airdrop", "may", "rumor", "potential"]):
        repost_count = random.randint(20, 150)
        first_poster_reputation = round(random.uniform(0.3, 0.6), 3)
        pattern = "low_engagement"
    else:
        repost_count = random.randint(5, 50)
        first_poster_reputation = round(random.uniform(0.4, 0.7), 3)
        pattern = "low_engagement"

    return {
        "source": "web3_social",
        "repost_count": repost_count,
        "first_poster_reputation": first_poster_reputation,
        "propagation_pattern": pattern,
        "details": {
            "lens": {"source": "lens", "post_count": repost_count // 2, "first_poster_reputation": first_poster_reputation},
            "farcaster": {"source": "farcaster", "cast_count": repost_count // 2, "first_poster_reputation": first_poster_reputation},
        },
    }


async def _mock_chain_reputation_retrieval(addr: str) -> dict:
    """
    Mock on-chain reputation retrieval.

    Generates plausible zScore / TraceRank data based on address hash.
    """
    await asyncio.sleep(random.uniform(0.1, 0.3))

    # Use the address to deterministically vary the score (demo purposes only)
    seed = hash(addr) % 100
    rng = random.Random(seed)
    zscore = rng.randint(30, 95)
    aura = rng.randint(20, 90)
    payment_backing = round(rng.uniform(0.0, 10.0), 2)

    return {
        "source": "chain_reputation",
        "zscore": zscore,
        "aura": aura,
        "payment_backing": payment_backing,
        "details": {
            "address": addr,
            "zscore_raw": {"zscore": zscore, "aura": aura, "payment_backing": payment_backing},
            "tracerank": {"rank": round(rng.uniform(0.1, 0.9), 2), "volume": round(rng.uniform(0.0, 100.0), 2)},
        },
    }
