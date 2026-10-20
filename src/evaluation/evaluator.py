import torch
from sklearn.metrics import classification_report, confusion_matrix

from src.training.metrics import compute_metrics


def load_checkpoint(model, path, device):
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    return model


def evaluate_on_test(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            metadata = batch["metadata"].to(device)
            labels = batch["label"].to(device)

            logits = model(images, metadata)
            loss = criterion(logits, labels)

            total_loss += loss.item()

            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader)
    metrics = compute_metrics(all_preds, all_labels)

    return avg_loss, metrics, all_preds, all_labels


def print_report(all_preds, all_labels, class_names):
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print(report)


def get_confusion_matrix(all_preds, all_labels):
    return confusion_matrix(all_labels, all_preds)
