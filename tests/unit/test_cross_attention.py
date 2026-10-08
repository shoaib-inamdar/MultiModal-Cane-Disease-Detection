import pytest
import torch

from src.models.fusion.cross_attention import CrossAttentionFusion

# ==========================================
# FIXTURE
# ==========================================


@pytest.fixture
def fusion():
    return CrossAttentionFusion()


# ==========================================
# TEST 1: OUTPUT SHAPE
# ==========================================


def test_output_shape(fusion):
    """
    Purpose:
    Output must be [B, 256] — ready for the classifier.
    """

    visual = torch.rand(2, 49, 768)
    meta = torch.rand(2, 1, 256)

    out, _ = fusion(visual, meta)

    assert out.shape == torch.Size([2, 256])


# ==========================================
# TEST 2: GRADIENT FLOWS
# ==========================================


def test_gradient_flows(fusion):
    """
    Purpose:
    Gradients must reach both visual and metadata inputs.
    """

    visual = torch.rand(2, 49, 768, requires_grad=True)

    meta = torch.rand(2, 1, 256, requires_grad=True)

    out, _ = fusion(visual, meta)

    out.sum().backward()

    # Gradient must reach visual features
    assert visual.grad is not None, "No gradient reached visual input"

    # Gradient must reach metadata features
    assert meta.grad is not None, "No gradient reached metadata input"


# ==========================================
# TEST 3: NO NaN IN OUTPUT
# ==========================================


def test_no_nan_in_output(fusion):
    """
    Purpose:
    Catch numerical instability early.
    """

    visual = torch.rand(2, 49, 768)
    meta = torch.rand(2, 1, 256)

    out, _ = fusion(visual, meta)

    assert not torch.isnan(out).any().item(), "Model output contains NaN values"
