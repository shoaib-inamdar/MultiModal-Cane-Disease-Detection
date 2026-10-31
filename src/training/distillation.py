import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    true_labels: torch.Tensor,
    temperature: float = 4.0,
    alpha: float = 0.5,
) -> tuple[torch.Tensor, float, float]:
    """
    Combined Knowledge Distillation loss (Hinton 2015).

    total_loss = alpha * CrossEntropy(student, hard_labels)
               + (1 - alpha) * T^2 * KL(student_soft || teacher_soft)

    Temperature T softens the teacher distribution so the student learns
    from the relative class probabilities, not just the argmax.

    Args:
        student_logits : [B, num_classes]
        teacher_logits : [B, num_classes]
        true_labels    : [B]  long tensor
        temperature    : softening factor (default 4.0)
        alpha          : weight for hard-label loss (default 0.5)

    Returns:
        total_loss   : combined scalar loss (differentiable)
        hard_loss    : float  (CrossEntropy component)
        distill_loss : float  (KL-divergence component)
    """
    # Soft targets
    teacher_probs = F.softmax(
        teacher_logits / temperature, dim=1
    )  # detached implicitly
    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)

    # KL divergence -- rescaled by T^2 to keep gradients in a consistent range
    distill = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
    distill = distill * (temperature**2)

    # Hard label loss
    hard = F.cross_entropy(student_logits, true_labels)

    total = alpha * hard + (1.0 - alpha) * distill

    return total, hard.item(), distill.item()


def train_student_one_epoch(
    teacher: nn.Module,
    student: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    temperature: float = 4.0,
    alpha: float = 0.5,
) -> float:
    """
    Run one epoch of knowledge distillation training.

    Teacher is frozen (eval mode, no grad).
    Student is trained with the combined distillation loss.

    Returns:
        average total loss over all batches
    """
    teacher.eval()
    student.train()

    total_loss = 0.0
    for batch in loader:
        images = batch["image"].to(device)
        metadata = batch["metadata"].to(device)
        labels = batch["label"].to(device)

        # Teacher forward -- no gradient
        with torch.no_grad():
            teacher_logits = teacher(images, metadata)

        # Student forward
        student_logits = student(images, metadata)

        loss, _, _ = distillation_loss(
            student_logits, teacher_logits, labels, temperature, alpha
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)
