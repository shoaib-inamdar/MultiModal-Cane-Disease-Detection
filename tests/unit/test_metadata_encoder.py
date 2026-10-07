import pytest
import torch

from src.models.encoders.metadata_encoder import MetadataEncoder


@pytest.fixture
def encoder() -> MetadataEncoder:
    return MetadataEncoder()


def test_output_shape(encoder: MetadataEncoder) -> None:
    batch_size = 4
    x = torch.rand(batch_size, 4)

    out = encoder(x)

    assert out.shape == torch.Size([batch_size, 1, 256])


def test_gradient_flows(encoder: MetadataEncoder) -> None:
    x = torch.rand(2, 4, requires_grad=True)

    out = encoder(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None