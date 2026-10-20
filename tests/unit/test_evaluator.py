import os
import tempfile

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.evaluation.evaluator import evaluate_on_test, load_checkpoint
from tests.unit.test_trainer import _DictDataset


def test_evaluate_on_test_returns_correct_keys():
    from src.models.sugarcane_classifier import SugarcaneClassifier

    model = SugarcaneClassifier(num_classes=4)
    loader = DataLoader(_DictDataset(n=4, num_classes=4), batch_size=2)
    criterion = nn.CrossEntropyLoss()

    avg_loss, metrics, preds, labels = evaluate_on_test(
        model, loader, criterion, device="cpu"
    )

    assert isinstance(avg_loss, float)
    assert "accuracy" in metrics
    assert len(preds) == len(labels) == 4


def test_load_checkpoint_roundtrip():
    from src.models.sugarcane_classifier import SugarcaneClassifier

    model = SugarcaneClassifier(num_classes=4)
    model.eval()

    img = torch.rand(1, 3, 224, 224)
    meta = torch.rand(1, 4)

    with torch.no_grad():
        out_before = model(img, meta)

    optimizer = torch.optim.Adam(model.parameters())
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "ckpt.pt")
        state = {
            "epoch": 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(state, path)

        model2 = SugarcaneClassifier(num_classes=4)
        model2 = load_checkpoint(model2, path, device="cpu")

        with torch.no_grad():
            out_after = model2(img, meta)

        assert torch.allclose(out_before, out_after)
