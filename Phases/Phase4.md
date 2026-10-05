# ⏳ PHASE 4 — Metadata MLP Encoder
**Time: 1–2 hours**

---

## 🎯 Goal
Build a small MLP that converts the 4 environmental numbers into a **256-dim vector**. This vector becomes the "environmental token" that image patches will query during Cross-Attention.

## 🤔 Why an MLP Encoder?

The metadata arrives as 4 raw numbers: `[Temperature, Humidity, Soil Moisture, Rainfall]`. These mean nothing to the Cross-Attention module — it needs rich high-dimensional vectors like the ones from the image backbone. The MLP **learns to embed** these 4 values into a 256-dim space.

After training, sensor readings that favour Red Rot (high temp + high humidity) will land in a very different region of that 256-dim space compared to readings that favour Smut (cool + wet soil). The Cross-Attention layer then uses that spatial relationship.

> 🧠 Think of it like translating weather data into the same language the image backbone speaks.

---

## 🪜 Step 4.1 — Create `src/models/encoders/metadata_encoder.py`

PSEUDOCODE — Do NOT copy. Write the Python yourself using this as a guide:

```
--- IMPORTS ---

import torch
import torch.nn as nn


--- CLASS: MetadataEncoder(nn.Module) ---

  Why this class?
    The 4 raw metadata values [T, RH, Sm, R] come in different units
    (°C, %, %, mm) and different numeric ranges.
    The MLP learns to project them into a unified 256-dim space
    that Cross-Attention can use alongside image patch tokens.


  __init__(self, input_dim=4, hidden_dim=128, output_dim=256, dropout=0.1):

    self.net = nn.Sequential(
        Linear(input_dim → 64),          ← expand: 4 raw values → 64 features
        ReLU(),
        Dropout(dropout),

        Linear(64 → hidden_dim),         ← 64 → 128, learn richer combinations
        ReLU(),
        Dropout(dropout),

        Linear(hidden_dim → output_dim)  ← 128 → 256 final embedding
    )

    Why three layers and not just one big Linear(4 → 256)?
      A single linear layer can only learn linear combinations.
      The relationship between weather and disease is non-linear
      (e.g., disease risk spikes above certain temperature thresholds, not
      proportionally). Stacking Linear + ReLU layers lets the network
      capture these threshold effects.

    Why Dropout?
      The training metadata for 300 dummy samples is random, so there is
      no real pattern to overfit. But when we later switch to real field
      data, Dropout prevents the MLP from memorising specific sensor
      readings and forces it to learn generalised weather patterns.


  ---


  forward(self, x):

    Args:
      x : Tensor of shape [B, 4]
              B = batch size
              4 = [Temperature, Humidity, Soil Moisture, Rainfall]

    Step 1: pass through the MLP
      out = self.net(x)         ← shape: [B, 256]

    Step 2: add a sequence dimension for Cross-Attention
      out = out.unsqueeze(1)    ← shape: [B, 1, 256]

    Why unsqueeze(1)?
      nn.MultiheadAttention (used in Phase 5) expects inputs as sequences
      with shape [B, seq_len, dim].
      The metadata becomes ONE token: seq_len = 1.
      All 49 image patch tokens will "attend to" this single environmental
      token — each patch asks: "Given the current weather, how should I
      interpret what I see in my region of the leaf?"

    return out    ← shape: [B, 1, 256]
```

---

## 🪜 Step 4.2 — Create `tests/unit/test_metadata_encoder.py`

PSEUDOCODE — Write 2 tests:

```
--- IMPORTS ---

import torch
import pytest
from src.models.encoders.metadata_encoder import MetadataEncoder


--- FIXTURE ---

  encoder():
    Return MetadataEncoder()

    Note: no pretrained weights here — MLP is always randomly initialised.
    We are only testing shapes and gradient flow, not learned behaviour.


---


  test_output_shape(encoder):

    Purpose:
      Verify the encoder produces exactly [B, 1, 256].
      If the unsqueeze(1) is missing, output will be [B, 256] (2D)
      and Cross-Attention will crash at runtime.

    B = 4
    x = torch.rand(B, 4)    ← 4 samples, each with 4 metadata values
    out = encoder(x)

    assert out.shape == torch.Size([B, 1, 256])


  ---


  test_gradient_flows(encoder):

    Purpose:
      Verify gradients travel back through all three Linear layers.
      If any layer is accidentally detached, the metadata encoder
      will never update its weights during training and will produce
      meaningless embeddings regardless of the weather input.

    x = torch.rand(2, 4, requires_grad=True)
    out = encoder(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
```

---

## ✅ Done When

```powershell
uv run pytest tests/unit/test_metadata_encoder.py -v
# 2 passed ✅
```

---

## ⚠️ Common Mistakes

| Mistake | Fix |
|---------|-----|
| Forgetting `unsqueeze(1)` | Output is `[B, 256]` → Cross-Attention crashes expecting `[B, seq, dim]` |
| Using `input_dim=3` | We have 4 metadata features — Temperature, Humidity, Soil Moisture, Rainfall |
| Missing `ReLU` between Linear layers | MLP collapses to a single linear transform — loses the ability to model thresholds |

---

## Key Tensor Shape After Phase 4

```
Metadata raw:    [B, 4]
      ↓
MetadataEncoder (MLP)
      ↓
Env token:       [B, 1, 256]   ← 1 environmental context token, 256-dim
```

All tensor shapes so far:

```
Image input:     [B, 3, 224, 224]   (Phase 2)
Swin patches:    [B, 49, 768]       (Phase 3)
Metadata raw:    [B, 4]             (Phase 2)
Env token:       [B, 1, 256]        (Phase 4)  ← YOU ARE HERE
```

---

> Say **"Phase 4 done"** when tests pass and I'll give you Phase 5.
