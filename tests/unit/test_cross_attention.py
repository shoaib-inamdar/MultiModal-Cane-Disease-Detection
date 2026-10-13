import pytest
import torch

from src.models.fusion.cross_attention import CrossAttentionFusion


@pytest.fixture
def fusion():
    return CrossAttentionFusion()


def test_output_shape(fusion):
    visual = torch.rand(2, 49, 768)
    meta = torch.rand(2, 1, 256)
    out = fusion(visual, meta)

    assert out.shape == torch.Size([2, 256])


def test_gradient_flows(fusion):
    visual = torch.rand(2, 49, 768, requires_grad=True)
    meta = torch.rand(2, 1, 256, requires_grad=True)
    out = fusion(visual, meta)
    out.sum().backward()

    assert visual.grad is not None
    assert meta.grad is not None


def test_no_nan_in_output(fusion):
    visual = torch.rand(2, 49, 768)
    meta = torch.rand(2, 1, 256)
    out = fusion(visual, meta)

    assert not torch.isnan(out).any()
