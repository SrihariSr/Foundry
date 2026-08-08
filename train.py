"""Dataset -> trained MLP + metrics."""

import random

import numpy as np
import torch
import torch.nn as nn

from featurise import FEATURE_DIM, featurise_batch

SEED = 0
HIDDEN_DIM = 128
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 1e-3
DEVICE = torch.device("cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def build_model(num_classes: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(FEATURE_DIM, HIDDEN_DIM),
        nn.ReLU(),
        nn.Linear(HIDDEN_DIM, num_classes),
    )


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """Rows are true classes, columns are predicted classes."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def per_class_metrics(cm: np.ndarray):
    """Precision, recall and support per class, derived from the confusion matrix."""
    num_classes = cm.shape[0]
    precision = np.zeros(num_classes, dtype=np.float64)
    recall = np.zeros(num_classes, dtype=np.float64)
    support = cm.sum(axis=1)

    for c in range(num_classes):
        tp = cm[c, c]
        predicted = cm[:, c].sum()
        actual = cm[c, :].sum()
        # A class the model never predicts has precision 0 by convention; the
        # confusion matrix printed alongside makes that case visible.
        precision[c] = tp / predicted if predicted > 0 else 0.0
        recall[c] = tp / actual if actual > 0 else 0.0

    return precision, recall, support


def train(X_train, y_train, X_test, y_test, num_classes):
    """Train the fixed MLP and return (model, metrics)."""
    Xtr = torch.from_numpy(np.asarray(X_train, dtype=np.float32)).to(DEVICE)
    ytr = torch.from_numpy(np.asarray(y_train, dtype=np.int64)).to(DEVICE)
    Xte = torch.from_numpy(np.asarray(X_test, dtype=np.float32)).to(DEVICE)
    yte = torch.from_numpy(np.asarray(y_test, dtype=np.int64)).to(DEVICE)

    model = build_model(num_classes).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    generator = torch.Generator().manual_seed(SEED)
    n = Xtr.shape[0]

    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = torch.randperm(n, generator=generator)
        epoch_loss = 0.0
        batches = 0

        for start in range(0, n, BATCH_SIZE):
            idx = order[start : start + BATCH_SIZE]
            optimiser.zero_grad()
            logits = model(Xtr[idx])
            loss = loss_fn(logits, ytr[idx])
            loss.backward()
            optimiser.step()
            epoch_loss += float(loss.detach())
            batches += 1

        if epoch % 5 == 0 or epoch == 1:
            print(f"epoch {epoch:2d}/{EPOCHS}  train loss {epoch_loss / batches:.4f}")

    model.eval()
    with torch.no_grad():
        preds = model(Xte).argmax(dim=1).cpu().numpy()

    y_true = yte.cpu().numpy()
    cm = confusion_matrix(y_true, preds, num_classes)
    precision, recall, support = per_class_metrics(cm)

    metrics = {
        "test_accuracy": float((preds == y_true).mean()),
        "precision": precision,
        "recall": recall,
        "support": support,
        "confusion_matrix": cm,
        "num_classes": num_classes,
    }
    return model, metrics


def print_metrics(metrics):
    num_classes = metrics["num_classes"]
    print(f"test accuracy: {metrics['test_accuracy']:.4f}")
    print()
    print("class  precision  recall  support")
    for c in range(num_classes):
        print(
            f"{c:5d}  {metrics['precision'][c]:9.4f}  "
            f"{metrics['recall'][c]:6.4f}  {metrics['support'][c]:7d}"
        )
    print()
    print("confusion matrix (rows = true, cols = predicted)")
    header = "        " + "".join(f"pred{c:<4d}" for c in range(num_classes))
    print(header)
    for c in range(num_classes):
        row = "".join(f"{v:<8d}" for v in metrics["confusion_matrix"][c])
        print(f"true{c:<4d}{row}")


# --- self-test: trivially separable synthetic data, no API involved ---------

MARKER = "zzurgent"
POSITIVE_RATE = 0.30


def make_vocab(rng, size=200):
    letters = "abcdefghijklmnopqrstuvwxyz"
    return ["".join(rng.choice(letters) for _ in range(rng.randint(3, 9))) for _ in range(size)]


def make_dataset(rng, vocab, count):
    texts, labels = [], []
    for _ in range(count):
        words = [rng.choice(vocab) for _ in range(rng.randint(8, 15))]
        if rng.random() < POSITIVE_RATE:
            words.insert(rng.randrange(len(words) + 1), MARKER)
            labels.append(1)
        else:
            labels.append(0)
        texts.append(" ".join(words))
    return texts, np.array(labels, dtype=np.int64)


def self_test():
    rng = random.Random(SEED)
    vocab = make_vocab(rng)

    train_texts, y_train = make_dataset(rng, vocab, 600)
    test_texts, y_test = make_dataset(rng, vocab, 200)

    print(f"train: {len(train_texts)} examples, {int(y_train.sum())} positive")
    print(f"test:  {len(test_texts)} examples, {int(y_test.sum())} positive")

    X_train = featurise_batch(train_texts)
    X_test = featurise_batch(test_texts)
    print(f"X_train {X_train.shape} {X_train.dtype}   X_test {X_test.shape} {X_test.dtype}")
    print()

    _, metrics = train(X_train, y_train, X_test, y_test, num_classes=2)
    print()
    print_metrics(metrics)
    print()

    accuracy_ok = metrics["test_accuracy"] > 0.95
    recall_ok = bool((metrics["recall"] > 0).all())
    print(f"ACCEPTANCE accuracy > 0.95: {accuracy_ok} (got {metrics['test_accuracy']:.4f})")
    print(f"ACCEPTANCE both classes non-zero recall: {recall_ok} "
          f"(got {[round(float(r), 4) for r in metrics['recall']]})")
    print("ACCEPTANCE CHECK:", "PASS" if (accuracy_ok and recall_ok) else "FAIL")


if __name__ == "__main__":
    self_test()
