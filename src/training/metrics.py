from sklearn.metrics import accuracy_score, f1_score


def compute_metrics(all_preds, all_labels):
    all_preds: list[int]
    all_labels: list[int]

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return {"accuracy": acc, "f1_macro": f1}
