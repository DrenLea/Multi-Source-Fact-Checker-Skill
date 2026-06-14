# Submission Checklist — Pharos + Anvita Flow Hackathon

## Core Deliverables

- [x] **Skill implementation** — Complete Python package in `src/`
- [x] **SKILL.md** — Structured description with input/output schema
- [x] **README.md** — Setup instructions, architecture diagram, usage examples
- [x] **Configuration system** — YAML-based with env var templating
- [x] **Mock mode** — Works without external API keys for evaluation
- [x] **Unit tests** — 5+ test cases covering trust/distrust/suspect/edge cases
- [x] **Demo video script** — 2-minute walkthrough in `demo_video_script.md`

## Technical Requirements

- [x] **Python 3.10+** compatible
- [x] **Three parallel evidence channels** (web2, web3 social, chain reputation)
- [x] **Asyncio-based parallelism** with 8-second total timeout
- [x] **Dynamic weight adjustment** based on evidence quality signals
- [x] **Conflict detection** when sources disagree significantly
- [x] **Grounding validation** (ML with keyword fallback)
- [x] **Graceful degradation** — `partial_data: true` when channels fail
- [x] **Structured logging** via loguru at INFO level
- [x] **Exception handling** on all external API calls with 5s per-call timeout

## Output Schema

- [x] `skill` field: `"multi_source_fact_checker_v1"`
- [x] `timestamp` in ISO-8601 UTC
- [x] `original_claim` preserved from input
- [x] `atomic_claims` list from decomposition
- [x] `overall_rating`: TRUST | SUSPECT | DISTRUST
- [x] `confidence`: float 0.0–1.0
- [x] `detailed_breakdown` with supported_by, contradicted_by, grounding_score
- [x] `evidence_trail` array with source/verdict/details per entry
- [x] `partial_data` boolean flag

## Input Features

- [x] Accepts `claim_text` (string, required)
- [x] Accepts `source_info` dict with optional `addr` and `url`
- [x] Rejects empty strings with clear error
- [x] Filters non-checkable claims (subjective, future predictions, questions)
- [x] Decomposes compound claims into atomic sub-claims

## Web3 Integrations

- [x] **Lens Protocol** — Client stub (real `lens-py-sdk` import path, mock data)
- [x] **Farcaster** — Client stub (real `farcaster-py` import path, mock data)
- [x] **zScore** — REST API stub with mock data
- [x] **TraceRank** — Simulated endpoint with mock data
- [x] **Propagation analysis** — normal vs astroturfing detection

## ML / AI Integration

- [x] **Loki framework** — Import path prepared, rule-based fallback active
- [x] **Paladin-mini** — HuggingFace loading with keyword-overlap fallback
- [x] **OpenAI** — Optional, config path prepared

## Code Quality

- [x] PEP8 compliant with docstrings on all public functions
- [x] Type hints throughout
- [x] Detailed inline comments explaining non-obvious logic
- [x] Modular architecture with clear separation of concerns
- [x] No hardcoded credentials (all via env vars or config)

## Testing

- [x] Claim normalization (whitespace, Unicode, edge cases)
- [x] Checkability filtering (subjective, questions, predictions)
- [x] Claim decomposition (simple and compound)
- [x] Verdict-to-score mapping
- [x] Evidence score extraction (web2, social, chain)
- [x] Conflict detection (no conflict and conflict cases)
- [x] Rating calculation (TRUST and DISTRUST thresholds)
- [x] Dynamic weight normalization (sums to 1.0)
- [x] Grounding validation (keyword overlap and empty evidence)
- [x] Full integration: trustworthy claim → TRUST
- [x] Full integration: false claim → DISTRUST/SUSPECT
- [x] Full integration: rumor → SUSPECT
- [x] Full integration: subjective → NOT_CHECKABLE
- [x] Full integration: empty input → ValueError
- [x] Full integration: output schema compliance
- [x] Full integration: source_info propagation

## Documentation

- [x] SKILL.md with complete schema
- [x] README.md with quick start
- [x] .env.example with all required vars
- [x] Inline code comments on all modules
- [x] Requirements pinned with minimum versions

## Hackathon-Specific

- [x] Pharos + Anvita Flow theme alignment (AI Agent fact-checking)
- [x] Multi-source approach (redundancy and cross-validation)
- [x] Web3 native (social graph + on-chain reputation, not just web2 APIs)
- [x] Hackathon-appropriate scope (ambitious but feasible within hackathon timeframe)
- [x] Demo-ready (mock mode for live evaluation without API keys)
