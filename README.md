# Multi-Source Fact Checker Skill

**Pharos + Anvita Flow Hackathon 2026 Submission**

A multi-source fact-checking pipeline for on-chain AI Agents. Given a textual
claim, the skill independently verifies it using three parallel evidence
channels and returns a structured **TRUST / SUSPECT / DISTRUST** rating.

## Architecture

```
Claim Text
   │
   ▼
Claim Processor    — normalize, filter non-checkable, decompose
   │
   ▼
Evidence Retriever ─── web2 (Google Fact Check + Loki)
   │               ─── web3_social (Lens + Farcaster)
   │               ─── chain_reputation (zScore + TraceRank)
   │  (asyncio.gather — all three in parallel, max 8s)
   ▼
Multi-Source Scorer — dynamic weighting, conflict detection, rating
   │
   ▼
Grounding Validator — Paladin-mini ML or keyword-overlap consistency
   │
   ▼
Final JSON Output (TRUST | SUSPECT | DISTRUST + confidence + evidence trail)
```

## Quick Start

### 1. Clone and set up

```bash
cd skill-verify-fact

# Create virtual environment
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your API keys (optional — mock mode works without them)
```

### 3. Run verification

```python
from src.main import verify_claim

result = verify_claim("Bitcoin is a cryptocurrency launched in 2009.")

print(result["overall_rating"])  # TRUST
print(result["confidence"])      # e.g. 0.85
print(result["evidence_trail"])
```

Or from the command line:

```bash
python -m src.main "Bitcoin is a cryptocurrency"
```

### 4. Run tests

```bash
pytest tests/ -v
```

## Configuration

All settings are in `config/`:

| File | Purpose |
|---|---|
| `api_config.yaml` | API endpoints, keys, mock toggle |
| `weights.yaml` | Source weights, rating thresholds, adjustments |

Set `mock.enabled: true` in `api_config.yaml` to use simulated data (no API keys needed).

## Environment Variables

Copy `.env.example` to `.env` and fill in the values. All API integrations are
optional — when a key is missing, that channel returns a neutral result or uses
mock data.

## Dependencies

- Python 3.10+
- `aiohttp` — Async HTTP for parallel API calls
- `loguru` — Structured logging
- `pyyaml` — Configuration
- `transformers` + `torch` — ML grounding (Paladin-mini)
- `pytest` + `pytest-asyncio` — Testing
