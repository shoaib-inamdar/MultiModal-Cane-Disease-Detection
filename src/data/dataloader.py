from torch.utils.data import DataLoader

from src.data.augmentation import (
    get_train_transforms,
    get_val_transforms,
)
from src.data.dataset import SugarcaneDataset


def get_dataloaders(config: dict) -> dict:
    synthetic = config["data"].get("synthetic", True)
    synthetic_metadata = config["data"].get("synthetic_metadata", True)

    batch_size = config["data"]["batch_size"]
    image_size = config["data"]["image_size"]

    num_workers = 0

    train_dataset = SugarcaneDataset(
        config=config,
        split="train",
        transform=get_train_transforms(image_size),
        synthetic=synthetic,
        synthetic_metadata=synthetic_metadata,
    )

    val_dataset = SugarcaneDataset(
        config=config,
        split="val",
        transform=get_val_transforms(image_size),
        synthetic=synthetic,
        synthetic_metadata=synthetic_metadata,
    )

    test_dataset = SugarcaneDataset(
        config=config,
        split="test",
        transform=get_val_transforms(image_size),
        synthetic=synthetic,
        synthetic_metadata=synthetic_metadata,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }