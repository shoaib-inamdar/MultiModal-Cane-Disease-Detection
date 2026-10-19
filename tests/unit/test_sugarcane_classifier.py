
import torch
import torch.nn as nn
import pytest
from src.models.sugarcane_classifier import SugarcaneClassifier


@pytest.fixture
def model():
    return SugarcaneClassifier(num_classes=6)

def test_output_shape(model):
    image=torch.rand(2,3,224,224)
    metadata=torch.rand(2,4)
    out=model(image,metadata)
    assert out.shape==torch.Size([2,6])

def test_gradient_flows(model):
    image=torch.rand(2,3,224,224,requires_grad=True)
    metadata=torch.rand(2,4,requires_grad=True)
    out=model(image,metadata)
    out.sum().backward()
    assert image.grad is not None
    assert metadata.grad is not None

def test_different_num_classes(model):
    m=SugarcaneClassifier(num_classes=10)
    img=torch.rand(1,3,224,224)
    met=torch.rand(1,4)
    out=m(img,met)
    assert out.shape==torch.Size([1,10])

