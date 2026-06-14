# multi_source_fact_checker

**Version:** 1.0.0
**Author:** Pharos + Anvita Flow Hackathon Team
**License:** MIT

## Description

A multi-source fact-checking Skill for on-chain AI Agents. Given a textual
claim, the skill independently verifies it by cross-referencing three parallel
evidence channels:

1. **Web2** — Traditional fact-checking APIs (Google Fact Check Tools, Loki)
2. **Web3 Social Graph** — Decentralized social protocols (Lens, Farcaster) with propagation analysis
3. **On-Chain Reputation** — zScore and TraceRank for wallet-level credibility scoring

The skill returns a structured `{TRUST | SUSPECT | DISTRUST}` rating,
a confidence score, and a full evidence trail.

## Input Schema

```json
{
  "claim_text": "string (required) — The claim to verify",
  "source_info": {
    "addr": "string (optional) — Ethereum wallet address for reputation lookup",
    "url": "string (optional) — Original URL where claim was found"
  }
}
```

## Output Schema

```json
{
  "skill": "multi_source_fact_checker_v1",
  "timestamp": "ISO-8601 UTC",
  "original_claim": "string",
  "atomic_claims": ["string"],
  "overall_rating": "TRUST | SUSPECT | DISTRUST",
  "confidence": 0.0-1.0,
  "detailed_breakdown": {
    "supported_by": ["list of source names"],
    "contradicted_by": ["list of source names"],
    "grounding_score": 0.0-1.0
  },
  "evidence_trail": [
    {
      "source": "string",
      "verdict": "string",
      "details": {}
    }
  ],
  "partial_data": false
}
```

## Required Environment Variables

| Variable | Description | Required |
|---|---|---|
| `GOOGLE_FACT_CHECK_API_KEY` | Google Fact Check Tools API key | No (mock fallback) |
| `LENS_API_KEY` | Lens Protocol API key | No (mock fallback) |
| `FARCASTER_HUB_URL` | Farcaster Hub gRPC URL | No (mock fallback) |
| `ZSCORE_RPC_URL` | zScore RPC endpoint | No (mock fallback) |
| `TRACERANK_ENDPOINT` | TraceRank API endpoint | No (mock fallback) |
| `OPENAI_API_KEY` | OpenAI API key (optional summary) | No |
| `LOG_LEVEL` | Logging level (default: INFO) | No |

Set these in a `.env` file (see `.env.example`).

## Usage Example

```python
from src.main import verify_claim

# Basic usage
result = verify_claim("Bitcoin is a cryptocurrency")
print(result["overall_rating"])  # TRUST
print(result["confidence"])      # e.g. 0.85

# With wallet context for on-chain reputation
result = verify_claim(
    "This project will airdrop tokens next month",
    source_info={"addr": "0x1234567890abcdef1234567890abcdef12345678"},
)
print(result["overall_rating"])  # SUSPECT
print(result["detailed_breakdown"]["grounding_score"])
```

## Configuration

- `config/api_config.yaml` — API endpoints and keys (uses `${ENV_VAR}` placeholders)
- `config/weights.yaml` — Source weights, adjustment rules, and rating thresholds

## Running Tests

```bash
# Install dependencies
pip install -r requirements.txt

# Run all tests
pytest tests/ -v

# Run a specific test
pytest tests/test_skill.py::test_trustworthy_claim -v
```

## Architecture

```
claim_text
  │
  ▼
claim_processor.py        ← normalize, checkability, decompose
  │
  ▼
evidence_retriever.py     ← 3 parallel retrievers (web2, social, chain)
  │
  ▼
multi_source_scorer.py    ← score extraction, dynamic weights, conflict resolution
  │
  ▼
grounding_validator.py    ← ML or keyword-based evidence-claim consistency
  │
  ▼
main.py: verify_claim()   ← assembles final output JSON
```

## Mock Mode

By default, all external API calls use mock data (controlled by `config/api_config.yaml`
→ `mock.enabled: true`). This allows testing and development without API keys.
Set to `false` and provide real keys in `.env` to use live APIs.

## Dependencies

See `requirements.txt` for the full list. Key dependencies:
- `loguru` — Structured logging
- `pyyaml` — Configuration loading
- `aiohttp` — Async HTTP client
- `transformers` + `torch` — ML grounding (Paladin-mini)
- `pytest` + `pytest-asyncio` — Testing
