import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.training.metrics import compute_metrics
from src.training.trainer import evaluate, train_one_epoch


def tiny_loader():
    images = torch.rand(4, 3, 224, 224)
    metadata = torch.rand(4, 4)
    labels = torch.randint(0, 4, (4,))

    dataset = TensorDataset(images, metadata, labels)

    return DataLoader(dataset, batch_size=2)


def tiny_model():
    from src.models.sugarcane_classifier import SugarcaneClassifier

    return SugarcaneClassifier(num_classes=4)


def test_train_one_epoch_returns_float():

    model = tiny_model()
    loader = tiny_loader()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    loss = train_one_epoch(model, loader, optimizer, criterion, device="cpu")

    assert isinstance(loss, float)
    assert not (loss != loss)


def test_evaluate_returns_metrics():
    model = tiny_model()
    loader = tiny_loader()
    criterion = nn.CrossEntropyLoss()

    avg_loss, metrics = evaluate(model, loader, criterion, device="cpu")

    assert isinstance(avg_loss, float)
    assert "accuracy" in metrics
    assert "f1_macro" in metrics


def test_compute_metrics_perfect():

    preds = [0, 1, 2, 3]
    labels = [0, 1, 2, 3]
    m = compute_metrics(preds, labels)

    assert m["accuracy"] == 1.0
