# ⏳ PHASE 5 — Cross-Attention Fusion ⭐
**Time: 3–4 hours — This is the core of your research**

---

## 🎯 Goal
Implement the **Cross-Attention Fusion** layer — the novel contribution of Objective 1. It connects what the image "sees" with what the environment "knows", producing context-aware visual features before classification.

## 🤔 The Big Idea

After Phases 3 and 4 you have two things:

```
SwinBackbone  →  Visual tokens:   [B, 49, 768]   ← 49 image patches
MetadataEncoder → Env token:     [B,  1, 256]   ← 1 weather token
```

The problem: they live in **different dimensions** (768 vs 256). You cannot do attention across mismatched dimensions.

The solution: **project** the visual tokens from 768 → 256, then run Cross-Attention:

```
After projection:
  Query (Q): [B, 49, 256]  ← image patches ask: "what disease am I?"
  Key   (K): [B,  1, 256]  ← weather token answers: "these are the conditions"
  Value (V): [B,  1, 256]  ← weather token provides context

Math:  Attention(Q, K, V) = softmax( Q · Kᵀ / √256 ) · V

Output: [B, 49, 256]   ← each patch is now "weather-aware"
Then:   mean over 49 patches → [B, 256]   ← single fused vector
```

> ⭐ **Research insight:** The attention weight for each patch tells you "how much did this region of the leaf care about the current weather?" — a directly visualisable, publishable finding.

---

## 🪜 Step 5.1 — Create `src/models/fusion/cross_attention.py`

PSEUDOCODE — Do NOT copy. Write the Python yourself:

```
--- IMPORTS ---

import torch
import torch.nn as nn


--- CLASS: CrossAttentionFusion(nn.Module) ---

  Why this class?
    This is the entire research contribution of Objective 1.
    Same image patch + different weather → different attention weights
    → different fused feature → (potentially) different diagnosis.


  __init__(self, visual_dim=768, metadata_dim=256, num_heads=8, dropout=0.1):

    Step A: Dimension alignment
      self.proj = nn.Linear(visual_dim, metadata_dim)
        ← Swin gives 768-dim tokens, metadata gives 256-dim tokens.
        ← Attention requires Q, K, V to share the same embed_dim.
        ← Project visual 768 → 256 to match.

    Step B: Multi-head cross-attention
      self.attention = nn.MultiheadAttention(
          embed_dim=metadata_dim,    ← 256
          num_heads=num_heads,       ← 8 heads run in parallel
          dropout=dropout,
          batch_first=True,          ← IMPORTANT: expects [B, seq, dim]
      )

      Why 8 heads?
        Each head specialises in a different kind of relationship.
        Head 1 might learn "lesion area vs temperature".
        Head 2 might learn "leaf edge vs humidity". Etc.
        The 8 outputs are concatenated → richer representation.

    Step C: Residual normalisation
      self.norm = nn.LayerNorm(metadata_dim)

      Why LayerNorm after attention?
        Prevents attention output values from growing too large or too
        small (gradient explosion / vanishing). Standard in all Transformers.

    Step D: Feed-forward network (FFN)
      self.ffn = nn.Sequential(
          nn.Linear(metadata_dim, metadata_dim * 2),   ← 256 → 512
          nn.ReLU(),
          nn.Dropout(dropout),
          nn.Linear(metadata_dim * 2, metadata_dim),   ← 512 → 256
      )

      Why FFN after attention?
        Attention mixes information across tokens (which patches relate to
        the environment?). FFN then transforms each token independently —
        adds non-linear expressiveness that pure attention lacks.

    Step E: Second layer norm (post-FFN)
      self.norm2 = nn.LayerNorm(metadata_dim)


  ---


  forward(self, visual_features, metadata_tokens):

    Args:
      visual_features  : [B, 49, 768]  ← from SwinBackbone
      metadata_tokens  : [B,  1, 256]  ← from MetadataEncoder

    Step 1: Project visual tokens from 768 → 256
      visual_proj = self.proj(visual_features)    ← [B, 49, 256]

    Step 2: Cross-Attention
      attn_out, attn_weights = self.attention(
          query=visual_proj,         ← [B, 49, 256]  image asks
          key=metadata_tokens,       ← [B,  1, 256]  weather answers
          value=metadata_tokens,     ← [B,  1, 256]  weather provides context
      )
      ← attn_out shape: [B, 49, 256]

    Step 3: Residual + LayerNorm (first)
      x = self.norm(visual_proj + attn_out)
        ← residual: preserves original visual info
        ← attn_out adds the weather-awareness on top

    Step 4: Feed-forward + residual + LayerNorm (second)
      x = self.norm2(x + self.ffn(x))

    Step 5: Pool 49 patch tokens into one vector
      fused = x.mean(dim=1)    ← [B, 256]

      Why mean pooling (not max or CLS token)?
        Mean pooling averages the context-aware signal across all 49
        leaf patches — no single patch dominates. Simpler and works well
        for classification.

    return fused    ← [B, 256]
```

---

## 🪜 Step 5.2 — Create `tests/unit/test_cross_attention.py`

PSEUDOCODE — Write 3 tests:

```
--- IMPORTS ---

import torch
import pytest
from src.models.fusion.cross_attention import CrossAttentionFusion


--- FIXTURE ---

  fusion():
    Return CrossAttentionFusion()


---


  test_output_shape(fusion):

    Purpose: Output must be [B, 256] — ready for the classifier

    visual = torch.rand(2, 49, 768)    ← 2 images, 49 patches, 768-dim
    meta   = torch.rand(2,  1, 256)    ← 2 env tokens, 256-dim
    out    = fusion(visual, meta)

    assert out.shape == torch.Size([2, 256])


  ---


  test_gradient_flows(fusion):

    Purpose: Gradients must reach both visual and metadata inputs

    visual = torch.rand(2, 49, 768, requires_grad=True)
    meta   = torch.rand(2,  1, 256, requires_grad=True)
    out    = fusion(visual, meta)
    out.sum().backward()

    assert visual.grad is not None
      Why? If None → backbone cannot be fine-tuned (frozen by accident).
    assert meta.grad is not None
      Why? If None → metadata encoder never learns from disease signal.


  ---


  test_no_nan_in_output(fusion):

    Purpose: Catch numerical instability early

    visual = torch.rand(2, 49, 768)
    meta   = torch.rand(2,  1, 256)
    out    = fusion(visual, meta)

    assert not torch.isnan(out).any()

    Why?
      NaN silently propagates through the whole network and breaks
      training without an obvious error message.
```

---

## ✅ Done When

```powershell
uv run pytest tests/unit/test_cross_attention.py -v
# 3 passed ✅
```

---

## ⚠️ Common Mistakes

| Mistake | Fix |
|---------|-----|
| Forgetting `self.proj` (Linear 768→256) | Attention crashes — Q dim ≠ K/V dim |
| `batch_first=False` (default) | Input/output will be `[seq, B, dim]` instead of `[B, seq, dim]` → wrong mean-pool axis |
| Missing residual connections | Gradients vanish in deep networks; residuals give them a shortcut |
| Mean pooling on wrong axis | `x.mean(dim=1)` pools over 49 patches ✅ — `dim=0` pools over batch ✗ |

---

## Key Tensor Shapes After Phase 5

```
visual_features:  [B, 49, 768]   (Phase 3)
        ↓  proj 768→256
visual_proj:      [B, 49, 256]
        ↓  cross-attention with metadata [B, 1, 256]
attn_out:         [B, 49, 256]
        ↓  residual + norm + FFN + norm
x:                [B, 49, 256]
        ↓  mean(dim=1)
fused:            [B, 256]        ← YOU ARE HERE
```

All shapes so far:

```
Image input:     [B, 3, 224, 224]   (Phase 2)
Swin patches:    [B, 49, 768]       (Phase 3)
Metadata raw:    [B, 4]             (Phase 2)
Env token:       [B, 1, 256]        (Phase 4)
Fused vector:    [B, 256]           (Phase 5) ← context-aware image+weather
```

---

> Say **"Phase 5 done"** when tests pass and I'll give you Phase 6.
