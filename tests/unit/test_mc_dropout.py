import torch

from src.evaluation.mc_dropout import compute_ece, mc_predict
from src.models.sugarcane_classifier import SugarcaneClassifier


def test_mc_predict_returns_correct_keys():
    """mc_predict must return all expected keys with valid value ranges."""
    model = SugarcaneClassifier(num_classes=4)
    image = torch.rand(1, 3, 224, 224)
    metadata = torch.rand(1, 4)

    result = mc_predict(model, image, metadata, device="cpu", num_passes=5)

    assert "pred_class" in result
    assert "confidence" in result
    assert "uncertainty" in result
    assert "mean_probs" in result
    assert len(result["mean_probs"]) == 4
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["uncertainty"] >= 0.0


def test_uncertainty_is_nonnegative():
    """Variance must always be >= 0."""
    model = SugarcaneClassifier(num_classes=4)
    image = torch.rand(1, 3, 224, 224)
    metadata = torch.rand(1, 4)

    result = mc_predict(model, image, metadata, device="cpu", num_passes=10)

    assert result["uncertainty"] >= 0.0


def test_ece_perfect_calibration():
    """100% confident and always correct -> ECE = 0.0."""
    confidences = [1.0, 1.0, 1.0, 1.0]
    correct_flags = [True, True, True, True]

    ece = compute_ece(confidences, correct_flags)

    assert ece == 0.0
