import numpy as np
import torch

from src.evaluation.gradcam import GradCAMPlusPlus, apply_heatmap
from src.models.sugarcane_classifier import SugarcaneClassifier


def test_gradcam_output_shape():
    """GradCAMPlusPlus must return a [224,224] heatmap in [0,1] and a valid class index."""
    model = SugarcaneClassifier(num_classes=4)
    cam = GradCAMPlusPlus(model)
    image = torch.rand(1, 3, 224, 224)
    metadata = torch.rand(1, 4)

    heatmap, pred_cls = cam(image, metadata, device="cpu")

    assert heatmap.shape == (224, 224)
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0
    assert 0 <= pred_cls < 4


def test_apply_heatmap_output():
    """apply_heatmap must return a [224,224,3] uint8 RGB image."""
    image_tensor = torch.rand(1, 3, 224, 224)
    heatmap = np.random.rand(224, 224).astype(np.float32)

    overlaid = apply_heatmap(image_tensor, heatmap)

    assert overlaid.shape == (224, 224, 3)
    assert overlaid.dtype == np.uint8
