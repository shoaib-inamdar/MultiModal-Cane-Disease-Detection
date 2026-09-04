# 🌾 SugarcaneAI — Phase 2: Dummy Dataset + DataLoader
## Scaffold Guide — (Do NOT copy — write Python yourself!)

> Read each scaffold → understand what it's asking → write the code yourself.
> Say **"Phase 2 done"** when all 4 tests pass.

---

## 🎯 Goal

Build a `SugarcaneDataset` that generates **fake/synthetic data** so you can test the full pipeline without waiting for real field images.

---

## 🤔 Why Synthetic First?

```
Stage A (Now)   → Synthetic random images + fake metadata → Test architecture ← YOU ARE HERE
Stage B (Next)  → Real Kaggle leaf images + fake metadata → Validate visual encoder
Stage C (Final) → Real Maharashtra photos + real DHT22 readings → Final results
```

When real data arrives, you change ONE thing: `synthetic=False`.

---

---

## 📄 File 1 of 7 — `configs/obj1_cross_attention.yaml`

> This is a YAML file, not Python. Use 2-space indentation. No tabs.

```
experiment:
  name:        ← string: "obj1_cross_attention_v1"
  seed:        ← integer: 42
  description: ← string: describe what this experiment does

data:
  num_classes:            ← integer: 4
  class_names:            ← list of 4 strings: Healthy, Red Rot, Grassy Shoot, Smut
  image_size:             ← integer: 224  (Swin-Tiny requires exactly 224)
  metadata_dim:           ← integer: 4  ([Temperature, Humidity, Soil Moisture, Rainfall])
  batch_size:             ← integer: 16  (safe for Colab T4 GPU)
  num_workers:            ← integer: 0  (MUST be 0 on Windows + Colab)
  num_synthetic_samples:  ← integer: 300
  synthetic:              ← boolean: true
  synthetic_metadata:     ← boolean: true

model:
  backbone:           ← string: "swin_tiny_patch4_window7_224"
  backbone_pretrained:← boolean: true
  backbone_out_dim:   ← integer: 768  (Swin-Tiny output, do not change)
  num_patches:        ← integer: 49   (7×7 spatial patches from Swin-Tiny)
  metadata_hidden_dims: ← list of 2 integers: [64, 128]
  metadata_out_dim:   ← integer: 256
  fusion_num_heads:   ← integer: 8
  fusion_dropout:     ← float: 0.1

training:
  num_epochs:     ← integer: 20
  learning_rate:  ← float: 0.0001
  weight_decay:   ← float: 0.0001
  patience:       ← integer: 5
  scheduler:      ← string: "cosine"

colab:
  use_drive:             ← boolean: true
  drive_checkpoint_path: ← string: "/content/drive/MyDrive/SugarcaneAI/checkpoints"

logging:
  use_wandb:     ← boolean: false  (set true when you have W&B API key)
  wandb_project: ← string: "sugarcane-ai-obj1"
```

**Verify:**
```powershell
uv run python -c "import yaml; cfg = yaml.safe_load(open('configs/obj1_cross_attention.yaml')); print(cfg['data']['class_names'])"
# Expected: ['Healthy', 'Red Rot', 'Grassy Shoot', 'Smut']
```

---

---

## 📄 File 2 of 7 — `src/utils/config.py`

```
--- IMPORTS ---
Import: pathlib.Path
Import: yaml

--- FUNCTION: load_config ---
load_config(path: str) -> dict:

  Convert path to a Path object

  If the file does NOT exist:
    Raise FileNotFoundError with message showing the full resolved path

  Open the file
  Parse it with yaml.safe_load()
  Return the parsed dict
```

**Verify:**
```powershell
uv run python -c "
from src.utils.config import load_config
cfg = load_config('configs/obj1_cross_attention.yaml')
print('OK:', cfg['data']['batch_size'])
"
# Expected: OK: 16
```

---

---

## 📄 File 3 of 7 — `src/utils/seed.py`

```
--- IMPORTS ---
Import: random
Import: numpy as np
Import: torch

--- FUNCTION: set_seed ---
set_seed(seed: int = 42) -> None:

  Set seed on: random module
  Set seed on: numpy
  Set seed on: torch (CPU)
  Set seed on: torch.cuda (GPU)
  Set seed on: torch.cuda for all GPUs (manual_seed_all)

  Set torch.backends.cudnn.deterministic = True   ← makes GPU ops reproducible
  Set torch.backends.cudnn.benchmark = False       ← disables auto-tuning
```

> 💡 **Why all 4?** `random`, `numpy`, and `torch` each have their own random number generator. Setting only one leaves the others unseeded — your results will still vary.

---

---

## 📄 File 4 of 7 — `src/data/augmentation.py`

```
--- IMPORTS ---
Import: albumentations as A
Import: ToTensorV2 from albumentations.pytorch

--- FUNCTION 1: get_train_transforms ---
get_train_transforms(image_size: int = 224) -> A.Compose:

  Return A.Compose of these transforms IN ORDER:
    1. Resize to (image_size, image_size)             ← always first
    2. HorizontalFlip with p=0.5
    3. VerticalFlip with p=0.2
    4. Rotate with limit=30 degrees, p=0.5
    5. ColorJitter — brightness=0.2, contrast=0.2,
                     saturation=0.2, hue=0.05, p=0.4
    6. CLAHE — clip_limit=2.0, tile_grid_size=(8,8), p=0.3
    7. GaussianBlur — blur_limit=(3,5), p=0.2
    8. Normalize — mean=[0.485, 0.456, 0.406]         ← ImageNet mean
                   std=[0.229, 0.224, 0.225]          ← ImageNet std
    9. ToTensorV2()                                   ← ALWAYS last

--- FUNCTION 2: get_val_transforms ---
get_val_transforms(image_size: int = 224) -> A.Compose:

  Return A.Compose of ONLY these 3 transforms (no random augmentation):
    1. Resize to (image_size, image_size)
    2. Normalize — same mean and std as above
    3. ToTensorV2()                                   ← ALWAYS last
```

> ⚠️ **ToTensorV2 must be LAST.** It converts numpy `[H, W, C]` uint8 → torch `[C, H, W]` float32. If placed earlier, other transforms break.

> 💡 **Why ImageNet mean/std?** Swin-Tiny was pretrained on ImageNet with those exact statistics. Normalizing differently will break the pretrained weights.

---

---

## 📄 File 5 of 7 — `src/data/dataset.py` ⭐

```
--- IMPORTS ---
Import: random
Import: Optional from typing
Import: numpy as np
Import: torch
Import: Dataset from torch.utils.data

--- CLASS: SugarcaneDataset(Dataset) ---

  CLASS CONSTANTS:
    CLASS_NAMES = ["Healthy", "Red Rot", "Grassy Shoot", "Smut"]
    NUM_CLASSES = 4

  --- __init__ ---
  __init__(self, config, split="train", transform=None,
           synthetic=True, synthetic_metadata=True,
           images_dir=None, metadata_csv=None):

    Assert split is one of: "train", "val", "test"
      ← if not, raise AssertionError with message

    Store as self: config, split, transform, synthetic,
                   synthetic_metadata
    Read from config and store as self:
      image_size ← config["data"]["image_size"]
      num_classes ← config["data"]["num_classes"]

    If synthetic is True:
      Call self._setup_synthetic(config)
    Else:
      Assert images_dir is not None
        ← message: "images_dir required when synthetic=False"
      Call self._setup_real(images_dir, metadata_csv)

  --- _setup_synthetic ---
  _setup_synthetic(self, config):

    total = config["data"]["num_synthetic_samples"]   ← 300

    Calculate split sizes:
      n_train = integer part of (total × 0.8)         ← 240
      n_val   = integer part of (total × 0.1)         ← 30
      n_test  = total − n_train − n_val               ← 30

    Store self.length based on self.split:
      "train" → n_train
      "val"   → n_val
      "test"  → n_test

    Generate ALL labels using numpy RandomState with seed=42:
      all_labels = RandomState(42).randint(0, self.num_classes, total)
      ← This gives reproducible random integers between 0 and 3

    Slice labels for this split:
      "train" → all_labels[ : n_train ]
      "val"   → all_labels[ n_train : n_train + n_val ]
      "test"  → all_labels[ n_train + n_val : ]

    Store slice as self.labels

  --- _setup_real ---
  _setup_real(self, images_dir, metadata_csv):

    Raise NotImplementedError with message:
      "Real dataset loading not yet implemented. Use synthetic=True for Stage A."

  --- __len__ ---
  __len__(self) -> int:

    Return self.length

  --- __getitem__ ---
  __getitem__(self, idx: int) -> dict:

    If self.synthetic:
      Return self._get_synthetic_sample(idx)
    Else:
      Return self._get_real_sample(idx)

  --- _get_synthetic_sample ---
  _get_synthetic_sample(self, idx: int) -> dict:

    image = torch.rand(3, self.image_size, self.image_size)
      ← random float tensor, values in [0, 1], shape [3, 224, 224]

    metadata = self._generate_synthetic_metadata()
      ← FloatTensor of shape [4], values in [0, 1]

    label = torch.tensor(self.labels[idx], dtype=torch.long)
      ← scalar LongTensor, value 0–3

    Return dict with keys: "image", "metadata", "label"

  --- _generate_synthetic_metadata ---
  _generate_synthetic_metadata(self) -> torch.FloatTensor:

    Generate 4 float values using random.uniform():
      temperature   = random.uniform(15.0, 45.0) / 50.0
      humidity      = random.uniform(30.0, 100.0) / 100.0
      soil_moisture = random.uniform(10.0, 90.0) / 100.0
      rainfall      = random.uniform(0.0, 200.0) / 300.0

    Return as torch.tensor([temperature, humidity, soil_moisture, rainfall],
                            dtype=torch.float32)

  --- _get_real_sample ---
  _get_real_sample(self, idx: int) -> dict:

    Raise NotImplementedError with message:
      "Stage B/C not yet implemented."
```

> 💡 **Why `RandomState(42)` for labels?** So every time you run the code, the same samples get the same labels. Without this, labels are random every run — your train/val split would be inconsistent.

> ⚠️ **Label dtype MUST be `torch.long`.** `CrossEntropyLoss` requires Long integers. Float labels will raise a RuntimeError.

---

---

## 📄 File 6 of 7 — `src/data/dataloader.py`

```
--- IMPORTS ---
Import: DataLoader from torch.utils.data
Import: get_train_transforms, get_val_transforms from src.data.augmentation
Import: SugarcaneDataset from src.data.dataset

--- FUNCTION: get_dataloaders ---
get_dataloaders(config: dict) -> dict:

  Read from config:
    synthetic          ← config["data"].get("synthetic", True)
    synthetic_metadata ← config["data"].get("synthetic_metadata", True)
    batch_size         ← config["data"]["batch_size"]
    image_size         ← config["data"]["image_size"]
    num_workers = 0    ← hardcode this, do NOT read from config

  Create 3 SugarcaneDataset objects:
    train_dataset: split="train", transform=get_train_transforms(image_size),
                   synthetic=synthetic, synthetic_metadata=synthetic_metadata
    val_dataset:   split="val",   transform=get_val_transforms(image_size), ...
    test_dataset:  split="test",  transform=get_val_transforms(image_size), ...

  Create 3 DataLoader objects:
    train_loader: dataset=train_dataset, batch_size=batch_size,
                  shuffle=True,  num_workers=0, pin_memory=False
    val_loader:   dataset=val_dataset,   batch_size=batch_size,
                  shuffle=False, num_workers=0, pin_memory=False
    test_loader:  dataset=test_dataset,  batch_size=batch_size,
                  shuffle=False, num_workers=0, pin_memory=False

  Return dict: {"train": train_loader, "val": val_loader, "test": test_loader}
```

---

---

## 📄 File 7 of 7 — `tests/unit/test_dataset.py`

```
--- IMPORTS ---
Import: torch
Import: load_config from src.utils.config
Import: SugarcaneDataset from src.data.dataset
Import: get_dataloaders from src.data.dataloader

--- CONSTANT ---
CONFIG_PATH = "configs/obj1_cross_attention.yaml"

--- TEST 1: test_dataset_length ---
test_dataset_length():

  Load config from CONFIG_PATH
  total = config["data"]["num_synthetic_samples"]   ← 300

  Create train, val, test datasets using SugarcaneDataset

  Assert len(train_dataset) == int(total × 0.8)    ← 240
  Assert len(val_dataset)   == int(total × 0.1)    ← 30
  Assert len(test_dataset)  == total − 240 − 30    ← 30
  Assert len(train) + len(val) + len(test) == total

--- TEST 2: test_sample_shapes ---
test_sample_shapes():

  Load config. Create train dataset. Get sample = dataset[0]

  Assert sample["image"].shape   == torch.Size([3, 224, 224])
  Assert sample["image"].dtype   == torch.float32
  Assert sample["metadata"].shape == torch.Size([4])
  Assert sample["metadata"].dtype == torch.float32
  Assert sample["label"].shape   == torch.Size([])    ← scalar, not [1]
  Assert sample["label"].dtype   == torch.long
  Assert 0 <= sample["label"].item() <= 3

--- TEST 3: test_metadata_range ---
test_metadata_range():

  Load config. Create train dataset.

  Loop i from 0 to 19 (first 20 samples):
    sample = dataset[i]
    Assert sample["metadata"].min().item() >= 0.0
    Assert sample["metadata"].max().item() <= 1.0

--- TEST 4: test_dataloader_batch ---
test_dataloader_batch():

  Load config.
  loaders = get_dataloaders(config)
  batch_size = config["data"]["batch_size"]

  Get one batch: batch = next(iter(loaders["train"]))

  Assert batch["image"].shape    == torch.Size([batch_size, 3, 224, 224])
  Assert batch["metadata"].shape == torch.Size([batch_size, 4])
  Assert batch["label"].shape    == torch.Size([batch_size])
  Assert batch["label"].dtype    == torch.long
```

---

## ✅ Phase 2 Success Check

```powershell
uv run pytest tests/unit/test_dataset.py -v
```

Expected:
```
test_dataset_length    PASSED   [ 25%]
test_sample_shapes     PASSED   [ 50%]
test_metadata_range    PASSED   [ 75%]
test_dataloader_batch  PASSED   [100%]
=================== 4 passed ===================
```

---

## ⚠️ Common Mistakes

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Missing `__init__.py` in `src/utils/` or `src/data/` | `ModuleNotFoundError` | Create empty `__init__.py` in every folder under `src/` |
| Wrong YAML indentation | `yaml.scanner.ScannerError` | 2 spaces only, no tabs |
| `num_workers > 0` | `RuntimeError: bootstrapping phase` | Always `num_workers=0` |
| Label as float not long | `RuntimeError: Expected scalar type Long` | `dtype=torch.long` |
| `ToTensorV2` not last | Shape mismatch errors | Always put `ToTensorV2()` at the end |
| pytest run from wrong directory | `FileNotFoundError: Config not found` | Run from project root |

---

*Come back and say **"Phase 2 done"** when all 4 tests pass! 🌾*
