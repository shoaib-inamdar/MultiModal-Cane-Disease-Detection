import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import SugarcaneDataset
from src.models.sugarcane_classifier import SugarcaneClassifier
from src.training.trainer import evaluate, train_one_epoch
from src.utils.config import load_config
from src.utils.seed import set_seed


def test_full_pipeline_synthetic():
    config = load_config("configs/obj1_cross_attention.yaml")
    set_seed(42)
    device = "cpu"

    train_ds = SugarcaneDataset(split="train", config=config)
    val_ds = SugarcaneDataset(split="val", config=config)
    train_loader = DataLoader(train_ds, batch_size=4)
    val_loader = DataLoader(val_ds, batch_size=4)

    model = SugarcaneClassifier(num_classes=config["data"]["num_classes"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

    assert isinstance(train_loss, float)
    assert not torch.isnan(torch.tensor(train_loss))

    val_loss, metrics = evaluate(model, val_loader, criterion, device)

    assert isinstance(val_loss, float)
    assert "accuracy" in metrics
    assert "f1_macro" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["f1_macro"] <= 1.0
