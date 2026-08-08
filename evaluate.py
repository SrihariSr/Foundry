"""Metrics, confusion matrix, learning curve."""

import json
import os
import random

import numpy as np
import torch

from featurise import featurise_batch
from train import SEED, print_metrics, train

SIZES = [50, 200, 1000, 4000]
DATA_DIR = "data"


def macro_f1(precision: np.ndarray, recall: np.ndarray) -> float:
    """Unweighted mean of per-class F1. A class with P+R == 0 contributes 0."""
    f1 = np.zeros_like(precision, dtype=np.float64)
    for c in range(len(precision)):
        denom = precision[c] + recall[c]
        f1[c] = 2.0 * precision[c] * recall[c] / denom if denom > 0 else 0.0
    return float(f1.mean())


def reset_seeds() -> None:
    """Re-seed before every train() so model init is identical across sizes.

    train.py seeds at import time only, so without this each successive call
    would start from different weights and dataset size would not be the only
    variable.
    """
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def learning_curve(train_examples, test_examples, num_classes, sizes=SIZES):
    """Train on nested subsets of increasing size, evaluate on one fixed test set."""
    X_train_all = featurise_batch([ex["text"] for ex in train_examples])
    y_train_all = np.array([ex["label"] for ex in train_examples], dtype=np.int64)
    X_test = featurise_batch([ex["text"] for ex in test_examples])
    y_test = np.array([ex["label"] for ex in test_examples], dtype=np.int64)

    # One fixed shuffle, then prefixes: subsets are nested and seed-stable.
    order = list(range(len(train_examples)))
    random.Random(SEED).shuffle(order)

    rows = []
    used_sizes = []
    for requested in sizes:
        n = min(requested, len(train_examples))
        if n < requested:
            print(f"note: requested n_train={requested} but only "
                  f"{len(train_examples)} examples available; using {n}")
        if n in used_sizes:
            print(f"note: skipping n_train={requested}, already covered by n_train={n}")
            continue
        used_sizes.append(n)

        idx = np.array(order[:n], dtype=np.int64)
        X_sub, y_sub = X_train_all[idx], y_train_all[idx]

        present = sorted(set(int(v) for v in y_sub))
        if len(present) < num_classes:
            missing = sorted(set(range(num_classes)) - set(present))
            print(f"note: subset n_train={n} contains no examples of class(es) {missing}")

        print(f"--- training on n_train={n} ---")
        reset_seeds()
        _, metrics = train(X_sub, y_sub, X_test, y_test, num_classes)
        rows.append({
            "n_train": n,
            "test_accuracy": metrics["test_accuracy"],
            "macro_f1": macro_f1(metrics["precision"], metrics["recall"]),
            "metrics": metrics,
        })
        print()

    return rows


def print_curve_table(rows):
    print("n_train  test_accuracy  macro_f1")
    for row in rows:
        print(f"{row['n_train']:7d}  {row['test_accuracy']:13.4f}  {row['macro_f1']:8.4f}")


def load_json(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Generate the corpus first (Phase 2 writes it)."
        )
    with open(path) as f:
        return json.load(f)


def main():
    spec = load_json(os.path.join(DATA_DIR, "spec.json"))
    train_examples = load_json(os.path.join(DATA_DIR, "train_A.json"))
    test_examples = load_json(os.path.join(DATA_DIR, "test_B.json"))

    print(f"task: {spec['task_name']}  num_classes: {spec['num_classes']}")
    for i, label in enumerate(spec["labels"]):
        print(f"class {i} = {label['name']}")
    print(f"train pool: {len(train_examples)}   test: {len(test_examples)}")
    print()

    rows = learning_curve(train_examples, test_examples, spec["num_classes"])

    print("=== LEARNING CURVE ===")
    print_curve_table(rows)
    print()
    print(f"=== FULL METRICS AT n_train={rows[-1]['n_train']} ===")
    print_metrics(rows[-1]["metrics"])


if __name__ == "__main__":
    main()
