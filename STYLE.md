# Code Style Guide

TLDR:
- simple, straightforward code
- fail fast - no defensive programming
- use einops, jaxtyping, and shape assertions for tensor shape clarity


## Core Principles

### Fail Fast (Negative Space Programming)
Code should fail immediately when assumptions are violated, preventing bugs from propagating.

```python
# BAD - silently handles unexpected state
def process_activations(acts):
    if acts is None:
        return torch.zeros(hidden_size)  # Hides the problem
    
# GOOD - fail immediately with clear error
def process_activations(acts):
    assert acts is not None, "Activations cannot be None"
    assert acts.dim() == 3, f"Expected 3D tensor, got {acts.dim()}D"
```

In practice:
- Use assertions liberally
- Use tensor shape assertions liberally
- Define types as tightly as possible
- Log warnings for edge cases
- Don't add soft failovers unless they're very much expected

## Type Annotations
- Don't add type annotations when they're redundant or obvious
- Use jaxtyping for tensor shapes (though for now we don't do runtime checking)

## Tensor Operations
- Try to use einops by default for clarity.
- Assert shapes liberally
- Document complex tensor manipulations

## Comments

Your first instinct should be: "How can I write this code so it's clear without comments?"

### Don't Write Obvious Comments

```python
# BAD
# get dataloader
dataloader = get_dataloader(config)

# make a tensor of all zeros
mask = torch.zeros(...)
```

### Do Comment Complex Logic

```python
# GOOD
# We need to mask out future positions for causal attention
# Upper triangular matrix excludes the diagonal (hence k=1)
causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
```

Watch: [Don't Write Comments (YouTube)](https://www.youtube.com/watch?v=Bf7vDBBOBUA)
