# Cascade Algorithm — Technical Deep Dive

The cascade decoder is the core routing engine of momo-kibidango. It routes
prompts through a 3-tier Claude model hierarchy, using heuristic confidence
scoring to minimise cost while maintaining quality.

## Overview

```
User Prompt
    │
    ▼
┌─────────────┐
│  HAIKU      │  ← cheapest, fastest (~$0.0003/query)
│  (Draft)    │
└─────┬───────┘
      │ response
      ▼
┌─────────────┐
│  CONFIDENCE │  ← score the draft (no API call, microseconds)
│  SCORER     │
└─────┬───────┘
      │
      ├── score ≥ 0.85 → ✅ ACCEPT Haiku result (done!)
      │
      ├── score 0.70–0.85 → escalate to Sonnet ↓
      │         │
      │    ┌────▼────────┐
      │    │  SONNET      │  ← refines Haiku's draft
      │    │  (Qualifier) │
      │    └────┬─────────┘
      │         │ refined response
      │         ▼
      │    ┌─────────────┐
      │    │  CONFIDENCE  │  ← re-score Sonnet's output
      │    │  SCORER      │
      │    └────┬─────────┘
      │         │
      │         ├── score ≥ 0.70 → ✅ ACCEPT Sonnet result
      │         │
      │         └── score < 0.70 → escalate to Opus ↓
      │
      └── score < 0.70 → skip Sonnet, go straight to Opus ↓
                │
           ┌────▼────────┐
           │  OPUS        │  ← full power, fresh generation
           │  (Target)    │
           └──────────────┘
                │
                ✅ ACCEPT (always — final tier)
```

## Confidence Scorer

The scorer evaluates a draft response **without any API calls**. It uses four
heuristic components, weighted by signal strength:

### Component Weights

| Component | Weight | Purpose |
|-----------|--------|---------|
| Complexity Match | 0.45 | Best signal — does the response match the prompt's complexity? |
| Coherence | 0.30 | Detects repetition, trailing off, incomplete outputs |
| Length Ratio | 0.15 | Response length vs. prompt length (tuned for short answers) |
| Self-Score | 0.10 | Optional: asks model to self-rate (disabled by default) |

Final score = weighted average of active components, normalised to [0.0, 1.0].

### Complexity Match (45% weight)

The strongest routing signal. Analyses the prompt for complexity indicators
and checks whether the response is proportionally substantive.

**Indicator keywords** (regex-matched):
`explain`, `analyze`, `compare`, `contrast`, `implement`, `design`, `architect`,
`debug`, `optimize`, `prove`, `derive`, `evaluate`, `synthesize`, `critique`,
`code`, `function`, `class`, `algorithm`, `theorem`, `equation`

**Code detection:**
Checks for code markers (`def`, `class`, `import`, `function`, `` ``` ``, etc.)
in both prompt and response. If code was requested but not provided, score drops
to 0.4.

**Scoring rules:**

| Prompt Complexity | Response Check | Score |
|-------------------|---------------|-------|
| 3+ indicators (complex) | ≥50 words (substantive) | 0.80 |
| 3+ indicators (complex) | <50 words (shallow) | 0.40 |
| 1-2 indicators (moderate) | — | 0.85 |
| 0 indicators (simple) | — | 0.95 |
| Code requested | Code absent in response | 0.40 |

### Coherence (30% weight)

Detects common failure modes in draft model outputs:

1. **Repetition detection** — sliding window bigram analysis. If any bigram
   repeats ≥3 times, applies a 0.30 penalty. Catches the "Haiku loop" problem
   where cheap models repeat phrases.

2. **Trailing off** — regex for `...`, `…`, `etc.`, `and so on` at end of
   response. Penalty: 0.15.

3. **Incomplete output** — checks if the response ends without terminal
   punctuation (`.`, `!`, `?`, `"`, `)`, `]`, `}`). Penalty: 0.10.

Penalties are cumulative. A clean response scores 1.0.

### Length Ratio (15% weight)

Compares response length to prompt length. Designed to avoid penalising
correct-but-short answers (e.g., factual lookups).

**Short-prompt bypass:** If the prompt is ≤15 words and the response is ≥5
words, the score is automatically 0.95. This prevents simple questions like
"What is the speed of light?" from being over-escalated due to a low
response/prompt ratio.

For longer prompts:
- `ratio = response_words / prompt_words`
- Ideal ratio: 5.0 (response is ~5x longer than prompt)
- Score tapers as ratio diverges from ideal
- Too short (ratio < 0.5): 0.30
- Too long (ratio > 50.0): 0.50

### Self-Score (10% weight, disabled by default)

Optional component that asks the draft model to self-rate its confidence on
a 1-10 scale. Costs extra tokens per request. When disabled (default), this
weight is redistributed to active components.

## Tier Thresholds

| Score Range | Tier | Action |
|-------------|------|--------|
| ≥ 0.85 | Haiku | Accept draft immediately |
| 0.70 – 0.85 | Sonnet | Refine the Haiku draft |
| < 0.70 | Opus | Full-power fresh generation |

These thresholds were tuned empirically using a 10-prompt test suite spanning
easy/medium/hard queries. See [Tuning Results](#tuning-results) below.

## Escalation Behaviour

### Haiku → Sonnet (Refinement)

When Haiku's draft scores in the Sonnet range (0.70–0.85), Sonnet receives
a **refinement prompt** that includes both the original prompt and Haiku's
draft:

```
The following response was generated for the prompt below, but may need
improvement. Please provide a better, more complete response.

Original prompt: {prompt}
Draft response: {haiku_draft}
Please provide an improved response:
```

This means Sonnet doesn't start from scratch — it builds on Haiku's work,
making it faster and more targeted.

### Second-Stage Escalation: Sonnet → Opus

After Sonnet produces its refined output, the confidence scorer runs again.
If Sonnet's output still scores below 0.70, the cascade escalates to Opus.

This catches edge cases where:
- Haiku's draft was borderline (scored 0.70–0.85)
- Sonnet's refinement didn't meaningfully improve it
- The prompt genuinely requires Opus-level reasoning

### Direct Opus Escalation

When Haiku's draft scores below 0.70, the cascade skips Sonnet entirely and
sends the **original prompt** (not the draft) directly to Opus. No draft
contamination — Opus gets a clean start.

## Cost Analysis

### Per-Request Cost by Path

| Path | API Calls | Typical Cost | Latency |
|------|-----------|-------------|---------|
| Haiku accepted | 1 (Haiku) | ~$0.0003 | ~1s |
| Sonnet refinement | 2 (Haiku + Sonnet) | ~$0.012 | ~3-5s |
| Sonnet → Opus escalation | 3 (Haiku + Sonnet + Opus) | ~$0.10 | ~8-12s |
| Direct Opus | 2 (Haiku + Opus) | ~$0.082 | ~6-10s |

### Expected Savings

Assuming typical real-world distribution (60% simple, 30% medium, 10% hard):

| Strategy | Monthly Cost (1000 queries) | Savings |
|----------|----------------------------|---------|
| All Opus | ~$82.00 | — |
| Cascade | ~$16.50 | **80%** |

Even with a conservative distribution (40/40/20), savings exceed 50%.

## Tuning Results

Validated April 2, 2026 with a 10-prompt test suite:

### Test Suite

| # | Prompt | Expected | Actual | Confidence | Cost | Result |
|---|--------|----------|--------|-----------|------|--------|
| E1 | "What is the speed of light?" | Haiku | Haiku | 96.7% | $0.0004 | ✅ |
| E2 | "What is 347 × 23?" | Haiku | Haiku | 93.3% | $0.0001 | ✅ |
| E3 | "Define photosynthesis" | Haiku | Haiku | 96.7% | $0.0002 | ✅ |
| M1 | Sieve of Eratosthenes (Python) | Sonnet | Sonnet | 74.9% | $0.012 | ✅ |
| M2 | TCP vs UDP comparison | Sonnet | Sonnet | 78.3% | $0.011 | ✅ |
| M3 | Race condition debugging | Sonnet | Sonnet | 75.7% | $0.012 | ✅ |
| H1 | Halting problem proof | Opus | Opus | 69.2% | $0.082 | ✅ |
| H2 | BFT consensus protocol | Opus | Opus | 69.2% | $0.100 | ✅ |
| H3 | Hindley-Milner type inference | Opus | Opus | 67.9% | $0.100 | ✅ |
| H4 | Mechanism design / game theory | Opus | Opus | 66.7% | $0.082 | ✅ |

**Score: 10/10 — Total test cost: $0.40 vs $0.82 all-Opus (51% savings)**

### Tuning History

Four rounds of threshold tuning were needed:

1. **v1 (0.90/0.70):** Simple facts over-escalated — LengthScore penalised
   short-but-correct answers (7/10)
2. **v2 (0.85/0.65):** Fixed LengthScore, rebalanced weights — hard prompts
   under-escalated because Sonnet accepted mediocre outputs (7/10)
3. **v3 (0.85/0.70):** Raised low threshold — 8/10, but Sonnet still caught
   two prompts that needed Opus
4. **v4 (0.85/0.70 + second-stage):** Added Sonnet→Opus re-evaluation — **10/10** ✅

## Source Files

| File | Purpose |
|------|---------|
| `src/momo_kibidango/core/cascade.py` | Cascade decoder — tier routing and escalation logic |
| `src/momo_kibidango/core/confidence.py` | Confidence scorer — heuristic scoring engine |
| `src/momo_kibidango/models/claude_client.py` | Claude API client — gateway + direct fallback |
| `src/momo_kibidango/proxy.py` | OpenAI-compatible proxy server (port 7780) |

## Design Principles

1. **No API calls for scoring** — all heuristic-based. The scorer runs in
   microseconds. The only API cost is actual model calls.

2. **Sonnet refines, Opus starts fresh** — Sonnet gets draft context for
   faster targeted improvement. Opus gets a clean prompt to avoid draft
   contamination.

3. **Fail upward** — every miss in routing escalates to a more capable model,
   never a cheaper one. Worst case is overspending, never under-delivering.

4. **Tunable without code changes** — thresholds and weights are constructor
   parameters, configurable at runtime.
