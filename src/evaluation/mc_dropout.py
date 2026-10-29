import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812


def enable_mc_dropout(model: torch.nn.Module) -> None:
    """
    Switch the model to eval mode (no batch-norm update)
    BUT keep Dropout layers active by setting them to train mode.
    This is the standard MC-Dropout setup.
    """
    model.eval()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()


def mc_predict(
    model: torch.nn.Module,
    image: torch.Tensor,
    metadata: torch.Tensor,
    device: str,
    num_passes: int = 50,
) -> dict:
    """
    Run num_passes stochastic forward passes and return mean prediction + uncertainty.

    Args:
        model      : SugarcaneClassifier (dropout kept ON via enable_mc_dropout)
        image      : [1, 3, 224, 224]
        metadata   : [1, 4]
        device     : "cuda" or "cpu"
        num_passes : number of stochastic passes (default 50)

    Returns dict:
        pred_class  : int
        confidence  : float in [0, 1]
        uncertainty : float >= 0  (higher = less certain)
        mean_probs  : np.ndarray [num_classes]
        all_probs   : np.ndarray [num_passes, num_classes]
    """
    enable_mc_dropout(model)
    image = image.to(device)
    metadata = metadata.to(device)

    all_probs = []
    with torch.no_grad():
        for _ in range(num_passes):
            logits = model(image, metadata)
            probs = F.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())

    stacked = np.stack(all_probs, axis=0)  # [num_passes, 1, num_classes]
    mean_probs = stacked.mean(axis=0)  # [1, num_classes]
    var_probs = stacked.var(axis=0)  # [1, num_classes]

    return {
        "pred_class": int(mean_probs.argmax(axis=1)[0]),
        "confidence": float(mean_probs.max(axis=1)[0]),
        "uncertainty": float(var_probs.max(axis=1)[0]),
        "mean_probs": mean_probs[0],
        "all_probs": stacked[:, 0],
    }


def batch_mc_predict(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: str,
    num_passes: int = 50,
) -> tuple[list, list, list, list]:
    """
    Run MC-Dropout over a full DataLoader.

    Returns:
        all_preds         : list[int]
        all_labels        : list[int]
        all_confidences   : list[float]
        all_uncertainties : list[float]
    """
    enable_mc_dropout(model)

    all_preds: list[int] = []
    all_labels: list[int] = []
    all_confidences: list[float] = []
    all_uncertainties: list[float] = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            metadata = batch["metadata"].to(device)
            labels = batch["label"]

            batch_probs = []
            for _ in range(num_passes):
                logits = model(images, metadata)
                probs = F.softmax(logits, dim=1)
                batch_probs.append(probs.cpu().numpy())

            stacked = np.stack(batch_probs, axis=0)  # [num_passes, B, num_classes]
            mean_p = stacked.mean(axis=0)
            var_p = stacked.var(axis=0)

            all_preds.extend(mean_p.argmax(axis=1).tolist())
            all_labels.extend(labels.tolist())
            all_confidences.extend(mean_p.max(axis=1).tolist())
            all_uncertainties.extend(var_p.max(axis=1).tolist())

    return all_preds, all_labels, all_confidences, all_uncertainties


def compute_ece(
    confidences: list[float],
    correct_flags: list[bool],
    n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error (ECE).

    For each confidence bin: |avg_confidence - avg_accuracy|.
    ECE = weighted average of those gaps.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(confidences)

    for b in range(n_bins):
        lo, hi = bins[b], bins[b + 1]
        in_bin = [i for i, c in enumerate(confidences) if lo < c <= hi]

        if not in_bin:
            continue

        bin_confidence = float(np.mean([confidences[i] for i in in_bin]))
        bin_accuracy = float(np.mean([float(correct_flags[i]) for i in in_bin]))
        ece += (len(in_bin) / n) * abs(bin_confidence - bin_accuracy)

    return ece
