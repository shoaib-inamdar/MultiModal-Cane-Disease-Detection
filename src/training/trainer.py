from pathlib import Path

import torch

from src.training.metrics import compute_metrics


def train_one_epoch(model, loader, optimizer, criterion, device):

    model.train()
    running_loss = 0.0

    for batch in loader:
        images = batch["image"].to(device)
        metadata = batch["metadata"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        logits = model(images, metadata)

        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def evaluate(model, loader, criterion, device):

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

    return avg_loss, metrics


def save_checkpoint(model, optimizer, epoch, path):

    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }

    torch.save(state, checkpoint_path)
    print(f"Checkpoint saved:{checkpoint_path}")
