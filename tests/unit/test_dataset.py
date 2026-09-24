import torch
from src.utils.config import load_config
from src.data.dataset import SugarcaneDataset
from src.data.dataloader import get_dataloaders
CONFIG_PATH = "configs/obj1_cross_attention.yaml"

def test_dataset_length():
    config = load_config(CONFIG_PATH)

    train_dataset = SugarcaneDataset(
        config,
        split="train",
        synthetic=True,
    )
    val_dataset = SugarcaneDataset(
        config,
        split="val",
        synthetic=True,
    )
    test_dataset = SugarcaneDataset(
        config,
        split="test",
        synthetic=True,
    )

    assert len(train_dataset) == 240
    assert len(val_dataset) == 30
    assert len(test_dataset) == 30

    assert len(train_dataset) + len(val_dataset) + len(test_dataset) == 300

def test_sample_shapes():
    config = load_config(CONFIG_PATH)

    dataset = SugarcaneDataset(
        config,
        split="train",
        synthetic=True,
    )

    sample = dataset[0]

    assert sample["image"].shape == torch.Size([3, 224, 224])
    assert sample["image"].dtype == torch.float32

    assert sample["metadata"].shape == torch.Size([4])
    assert sample["metadata"].dtype == torch.float32

    assert sample["label"].shape == torch.Size([])
    assert sample["label"].dtype == torch.long

    assert 0 <= sample["label"].item() <= 3


def test_metadata_range():
    config = load_config(CONFIG_PATH)

    dataset = SugarcaneDataset(
        config,
        split="train",
        synthetic=True,
    )

    for i in range(20):
        sample = dataset[i]
        metadata = sample["metadata"]

        assert metadata.min() >= 0.0
        assert metadata.max() <= 1.0


def test_dataloader_batch():
    config = load_config(CONFIG_PATH)

    loaders = get_dataloaders(config)

    batch = next(iter(loaders["train"]))
    assert batch["image"].shape == torch.Size([16, 3, 224, 224])
    assert batch["metadata"].shape == torch.Size([16, 4])
    assert batch["label"].shape == torch.Size([16])

    assert batch["label"].dtype == torch.long