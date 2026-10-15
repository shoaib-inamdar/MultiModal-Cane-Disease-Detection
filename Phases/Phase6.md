# PHASE 6 — SugarcaneClassifier (Full Model Assembly)
**Time: 1–2 hours**

---

## Goal

Wire all three modules you have built into one callable `nn.Module`:

```
Image [B, 3, 224, 224]  --- SwinBackbone ──────────────► visual_features [B, 49, 768]
                                                                   |
Metadata [B, 4]  --------- MetadataEncoder ────────────► meta_tokens [B, 1, 256]
                                                                   |
                                              CrossAttentionFusion ▼
                                                          fused [B, 256]
                                                                   |
                                              nn.Linear(256, num_classes) ▼
                                                          logits [B, num_classes]
```

After this phase you have a single `model(image, metadata)` call that produces disease class logits — the complete Objective 1 forward pass.

---

## Step 6.1 — Create `src/models/sugarcane_classifier.py`

PSEUDOCODE — Write the Python yourself:

```
--- IMPORTS ---

import torch
import torch.nn as nn

from src.models.backbone.swin_backbone      import SwinBackbone
from src.models.encoders.metadata_encoder   import MetadataEncoder
from src.models.fusion.cross_attention      import CrossAttentionFusion


--- CLASS: SugarcaneClassifier(nn.Module) ---

  Why this class?
    Clean, single entry-point for training, evaluation, and deployment.
    Keeps each module's responsibility separate (backbone / encoder / fusion / head).


  __init__(self, num_classes=6):

    Step A: Visual backbone
      self.backbone = SwinBackbone()
        Accepts [B, 3, 224, 224], outputs [B, 49, 768]

    Step B: Metadata encoder
      self.metadata_encoder = MetadataEncoder()
        Accepts [B, 4], outputs [B, 1, 256]

    Step C: Cross-attention fusion
      self.fusion = CrossAttentionFusion()
        Accepts visual [B, 49, 768] + meta [B, 1, 256], outputs [B, 256]

    Step D: Classification head
      self.classifier = nn.Linear(256, num_classes)
        Accepts [B, 256], outputs logits [B, num_classes]

      Why a single Linear with no softmax?
        During training we use nn.CrossEntropyLoss which applies
        log-softmax internally. Raw logits are numerically more stable.


  ---


  forward(self, image, metadata):

    Args:
      image    : [B, 3, 224, 224]   <- standard normalised RGB tensor
      metadata : [B, 4]             <- [temp, humidity, soil_moisture, rainfall]

    Step 1: Extract visual patch tokens
      visual_features = self.backbone(image)         <- [B, 49, 768]

    Step 2: Encode environmental context
      meta_tokens = self.metadata_encoder(metadata)  <- [B, 1, 256]

    Step 3: Fuse image + environment
      fused = self.fusion(visual_features, meta_tokens)  <- [B, 256]

    Step 4: Classify
      logits = self.classifier(fused)                <- [B, num_classes]

    return logits
```

---

## Step 6.2 — Create `tests/unit/test_sugarcane_classifier.py`

PSEUDOCODE — Write 3 tests:

```
--- IMPORTS ---

import torch
import pytest
from src.models.sugarcane_classifier import SugarcaneClassifier


--- FIXTURE ---

  @pytest.fixture
  model():
    return SugarcaneClassifier(num_classes=6)


---

  test_output_shape(model):

    Purpose: logits must be [B, num_classes]

    image    = torch.rand(2, 3, 224, 224)
    metadata = torch.rand(2, 4)
    out      = model(image, metadata)

    assert out.shape == torch.Size([2, 6])


  ---


  test_gradient_flows(model):

    Purpose: backward pass must not crash; gradients reach the inputs

    image    = torch.rand(2, 3, 224, 224, requires_grad=True)
    metadata = torch.rand(2, 4, requires_grad=True)
    out      = model(image, metadata)
    out.sum().backward()

    assert image.grad is not None
    assert metadata.grad is not None


  ---


  test_different_num_classes(model):

    Purpose: the head adapts to any number of classes (e.g. a future 10-class dataset)

    m   = SugarcaneClassifier(num_classes=10)
    img = torch.rand(1, 3, 224, 224)
    met = torch.rand(1, 4)
    out = m(img, met)

    assert out.shape == torch.Size([1, 10])
```

---

## Done When

```powershell
uv run pytest tests/unit/test_sugarcane_classifier.py -v
# 3 passed

uv run pytest tests/unit/ -v
# All 19 pass
```

---

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Forgetting `super().__init__()` | AttributeError when assigning sub-modules |
| Calling `self.backbone(image)` before defining it in `__init__` | AttributeError at runtime |
| Applying `torch.softmax` in `forward` | Loss explodes — CrossEntropyLoss expects raw logits |
| Wrong argument order in `forward` | `model(metadata, image)` silently produces bad results |

---

## Key Tensor Shapes After Phase 6

```
Input image:      [B, 3, 224, 224]
        v  SwinBackbone
visual_features:  [B, 49, 768]
        v  (+ MetadataEncoder output)
meta_tokens:      [B,  1, 256]
        v  CrossAttentionFusion
fused:            [B, 256]
        v  nn.Linear(256, num_classes)
logits:           [B, num_classes]   <- YOU ARE HERE
```

---

> Say "Phase 6 done" when tests pass and I will give you Phase 7.
