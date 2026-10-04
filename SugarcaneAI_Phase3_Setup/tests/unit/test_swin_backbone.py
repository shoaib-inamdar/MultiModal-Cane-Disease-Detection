import pytest
import torch

from src.models.backbone.swin_backbone import SwinBackbone


@pytest.fixture(scope="module")
def backbone():
    # pretrained=False skips the ~112MB weight download so CI runs instantly.
    # We're only checking shapes and gradient flow here, not accuracy.
    return SwinBackbone(pretrained=False)


def test_output_shape(backbone):
    batch_size = 2
    x = torch.rand(batch_size, 3, 224, 224)

    out = backbone(x)

    assert out.shape == torch.Size([batch_size, 49, 768])


def test_gradient_flows(backbone):
    x = torch.rand(1, 3, 224, 224, requires_grad=True)

    out = backbone(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None


def test_no_classification_output(backbone):
    x = torch.rand(1, 3, 224, 224)

    out = backbone(x)

    assert out.dim() == 3
