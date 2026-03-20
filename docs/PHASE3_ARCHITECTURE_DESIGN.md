# Phase 3: 3-Model Pyramid Architecture Design

**Date:** March 19, 2026  
**Time:** 9:05 PM EDT

## Overview

Building on the successful 2-model implementation (1.92x speedup), Phase 3 introduces a 3-model hierarchical verification system inspired by Google Research's PyramidSD paper.

## Architecture

```
┌─────────────────────────────┐
│   Qwen2-7B-4bit (Target)    │ ← Stage 3: Final Verification
│      Highest Quality        │    (Accurate, 4GB VRAM)
└─────────────────────────────┘
              ↑
      Stage 2 Verification
              ↑
┌─────────────────────────────┐
│     Phi-2 2.7B (Qualifier)   │ ← Stage 2: Quality Filter  
│      Medium Quality          │    (Balanced, 1.5-2GB VRAM)
└─────────────────────────────┘
              ↑
      Stage 1 Verification
              ↑
┌─────────────────────────────┐
│   Qwen2-0.5B (Draft)        │ ← Stage 1: Fast Generation
│      Fast Generation         │    (Ultra-fast, 2-3GB VRAM)
└─────────────────────────────┘
```

## Token Flow Algorithm

### 1. Draft Generation (Stage 1)
- **Model:** Qwen2-0.5B 
- **Goal:** Generate 5-8 tokens rapidly
- **Speed:** ~500 tokens/sec on Apple Silicon
- **Output:** List of draft tokens + probabilities

### 2. Qualifier Verification (Stage 2)
- **Model:** Phi-2 2.7B
- **Goal:** Filter out obviously wrong tokens
- **Acceptance Threshold:** 0.10 (more lenient)
- **Process:**
  - Check each draft token's probability
  - Accept if P(token|context) > 0.10
  - Reject and stop chain if below threshold
  - Pass accepted tokens to Stage 3

### 3. Target Verification (Stage 3)
- **Model:** Qwen2-7B-4bit (same as Phase 2)
- **Goal:** Final quality check
- **Acceptance Threshold:** 0.03 (strict, same as Phase 2)
- **Process:**
  - Verify tokens accepted by qualifier
  - Final accept/reject decision
  - If rejected, sample from target distribution

## Memory Allocation Plan

| Component | Memory Usage | Notes |
|-----------|--------------|-------|
| Draft (Qwen2-0.5B) | 2.0 GB | FP16 weights |
| Qualifier (Phi-2) | 2.5 GB | FP16 weights |
| Target (Qwen2-7B) | 4.0 GB | 4-bit quantized |
| KV Cache (combined) | 1.5 GB | Shared across models |
| PyTorch overhead | 1.0 GB | Tensors, gradients |
| **Total** | **11.0 GB** | Within 12GB limit ✅ |

## Acceptance Logic Design

### Two-Stage Filtering

```python
def hierarchical_verify(draft_tokens, draft_probs):
    # Stage 1 → 2: Draft to Qualifier
    stage1_accepted = []
    for token, prob in zip(draft_tokens, draft_probs):
        qualifier_prob = qualifier_model.get_prob(token)
        if qualifier_prob > STAGE1_THRESHOLD:  # 0.10
            stage1_accepted.append(token)
        else:
            break  # Stop at first rejection
    
    # Stage 2 → 3: Qualifier to Target
    final_accepted = []
    for token in stage1_accepted:
        target_prob = target_model.get_prob(token)
        if target_prob > STAGE2_THRESHOLD:  # 0.03
            final_accepted.append(token)
        else:
            # Sample from target and stop
            new_token = target_model.sample()
            final_accepted.append(new_token)
            break
    
    return final_accepted
```

## Expected Performance

### Speedup Calculation
- **Draft speed:** ~500 tok/s (0.5B model)
- **Qualifier speed:** ~50 tok/s (2.7B model)
- **Target speed:** ~12 tok/s (7B model)

With expected acceptance rates:
- Stage 1→2: 85% (draft similar to Qwen family)
- Stage 2→3: 90% (qualifier aligns well with target)
- Combined: 76.5% acceptance

**Expected throughput:** 23-26 tokens/sec (1.85-2.1x speedup)

## Fallback Strategy

```python
try:
    # Try 3-model pipeline
    return three_model_generate(prompt)
except (OutOfMemoryError, RuntimeError):
    try:
        # Fallback to proven 2-model
        return two_model_generate(prompt)
    except:
        # Ultimate fallback to single model
        return single_model_generate(prompt)
```

## Implementation Plan

1. **Load Models** (30 min)
   - Load all three models with proper device placement
   - Verify memory usage stays within bounds
   - Test basic inference on each

2. **Token Flow** (1 hour)
   - Implement hierarchical verification
   - Add acceptance threshold tuning
   - Ensure proper token passing between stages

3. **Optimization** (1 hour)
   - Dynamic speculation length based on acceptance
   - Batch processing where possible
   - Memory-efficient KV cache sharing

4. **Integration** (30 min)
   - Update OpenClaw API endpoints
   - Add configuration flags
   - Maintain backward compatibility

## Risk Mitigation

1. **Memory Overflow**
   - Monitor after each model load
   - Use `torch.cuda.empty_cache()` aggressively
   - Implement staged loading if needed

2. **Poor Acceptance Rates**
   - Start with conservative thresholds
   - Log detailed acceptance statistics
   - Tune based on empirical results

3. **Model Incompatibility**
   - Phi-2 uses different tokenizer
   - May need token mapping layer
   - Fallback to all-Qwen2 models if needed

## Success Criteria

- ✅ Speedup: 1.85-2.1x (23-26 tok/s)
- ✅ Memory: <12GB sustained
- ✅ Quality: Zero degradation
- ✅ Acceptance: 70%+ combined
- ✅ Fallback: Working 3→2→1 chain
- ✅ Documentation: Complete

## Next Step

Proceed to Task 2: Implementation of the 3-model pipeline in `src/speculative_3model.py`