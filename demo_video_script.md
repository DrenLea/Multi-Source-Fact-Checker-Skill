# Demo Video Script (2 minutes)

## Scene 1: Problem Introduction (0:00–0:20)

**Visuals:** Show a feed of AI agents on-chain making claims — some true, some
false, some speculative.

> "AI agents in Web3 generate and amplify thousands of claims every day.
> But how do we know which ones to trust?  Multi-Source Fact Checker solves
> this by cross-referencing three independent evidence channels in real time."

## Scene 2: Architecture Overview (0:20–0:50)

**Visuals:** Animated diagram showing the pipeline flow.

> "When a claim enters the system, it's first normalized and decomposed into
> atomic sub-claims. Then three parallel retrievers fire simultaneously:
> traditional fact-checking APIs like Google Fact Check Tools; Web3 social
> graphs from Lens and Farcaster; and on-chain reputation from zScore and
> TraceRank.  All of this completes in under 8 seconds."

## Scene 3: Live Demo — True Claim (0:50–1:10)

**Visuals:** Terminal showing a Python script calling `verify_claim("Bitcoin is
a cryptocurrency launched in 2009.")`.

```python
>>> from src.main import verify_claim
>>> result = verify_claim("Bitcoin is a cryptocurrency launched in 2009.")
>>> print(result["overall_rating"])
TRUST
>>> print(result["confidence"])
0.87
```

> "A well-known factual claim returns TRUST with high confidence.  The
> evidence trail shows support from Google Fact Check, high-reputation first
> posters, and solid on-chain signals."

## Scene 4: Live Demo — False/Manipulative Claim (1:10–1:35)

**Visuals:** Terminal showing a false claim being flagged.

```python
>>> result = verify_claim("Ethereum will crash to zero and become worthless.")
>>> print(result["overall_rating"])
DISTRUST
>>> print(result["detailed_breakdown"]["grounding_score"])
0.12
```

> "A demonstrably false or fear-mongering claim is flagged as DISTRUST.  The
> system detects the astroturfing propagation pattern and low first-poster
> reputation — key signals of coordinated disinformation."

## Scene 5: Live Demo — Rumor / Uncertain (1:35–1:50)

**Visuals:** A SUSPECT rating for an unconfirmed airdrop rumor.

> "When a claim cannot be verified — like an unconfirmed airdrop rumor — the
> system correctly returns SUSPECT with a moderate confidence score, and
> marks `partial_data: false` transparently."

## Scene 6: Closing (1:50–2:00)

**Visuals:** Pharos + Anvita Flow logos, link to GitHub repo.

> "Multi-Source Fact Checker: bringing verifiable truth to on-chain AI agents.
> Built for Pharos + Anvita Flow Hackathon 2026.  Link in the description."
