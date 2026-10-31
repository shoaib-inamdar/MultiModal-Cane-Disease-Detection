import torch

from src.models.student_classifier import StudentClassifier
from src.models.sugarcane_classifier import SugarcaneClassifier
from src.training.distillation import distillation_loss


def test_distillation_loss_values():
    """Combined loss must be a non-negative float; hard and soft components must be >= 0."""
    student_logits = torch.rand(8, 4)
    teacher_logits = torch.rand(8, 4)
    labels = torch.randint(0, 4, (8,))

    loss, hard, soft = distillation_loss(student_logits, teacher_logits, labels)

    assert isinstance(loss.item(), float)
    assert loss.item() >= 0.0
    assert hard >= 0.0
    assert soft >= 0.0


def test_student_output_shape():
    """StudentClassifier forward must return [B, num_classes] logits."""
    model = StudentClassifier(num_classes=4)
    image = torch.rand(2, 3, 224, 224)
    metadata = torch.rand(2, 4)

    logits = model(image, metadata)

    assert logits.shape == (2, 4)


def test_student_smaller_than_teacher():
    """StudentClassifier must have at least 40% fewer params than SugarcaneClassifier."""
    teacher = SugarcaneClassifier(num_classes=4)
    student = StudentClassifier(num_classes=4)

    teacher_params = sum(p.numel() for p in teacher.parameters())
    student_params = sum(p.numel() for p in student.parameters())

    # Student should be < 60% of teacher size (>= 40% smaller)
    assert student_params < teacher_params * 0.6
