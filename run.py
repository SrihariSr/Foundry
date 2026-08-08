"""Orchestrator: plain English need -> deployable single-file classifier."""

import argparse
import json
import os
import random
import re

import numpy as np
import torch

from emit import casting_path, emit
from featurise import featurise_batch
from generate import generate
from spec import make_spec
from train import SEED, print_metrics, train

DATA_DIR = "data"
DEFAULT_N_TRAIN = 4000
DEFAULT_N_TEST = 100


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    if not slug:
        raise ValueError(f"task_name {name!r} slugified to an empty string")
    return slug


def write_json(path, payload):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def run(need, n_train, n_test):
    print("[1/5] spec")
    spec = make_spec(need)
    slug = slugify(spec["task_name"])
    task_dir = os.path.join(DATA_DIR, slug)
    os.makedirs(task_dir, exist_ok=True)
    write_json(os.path.join(task_dir, "spec.json"), spec)
    print(f"      task_name: {spec['task_name']}  ->  {task_dir}/")
    print(f"      num_classes: {spec['num_classes']}  personas: {len(spec['generation_personas'])}")
    for i, label in enumerate(spec["labels"]):
        print(f"      class {i} = {label['name']}")
    print()

    print(f"[2/5] generate test set (n={n_test}, seed_tag=B)")
    test_examples = generate(spec, n_test, seed_tag="B", out_dir=task_dir)
    write_json(os.path.join(task_dir, "test_B.json"), test_examples)
    print()

    print(f"[3/5] generate train set (n={n_train}, seed_tag=A)")
    train_examples = generate(spec, n_train, seed_tag="A", out_dir=task_dir)
    write_json(os.path.join(task_dir, "train_A.json"), train_examples)
    print()

    print("[4/5] train")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    X_train = featurise_batch([ex["text"] for ex in train_examples])
    y_train = np.array([ex["label"] for ex in train_examples], dtype=np.int64)
    X_test = featurise_batch([ex["text"] for ex in test_examples])
    y_test = np.array([ex["label"] for ex in test_examples], dtype=np.int64)
    print(f"      X_train {X_train.shape}   X_test {X_test.shape}")

    model, metrics = train(X_train, y_train, X_test, y_test, spec["num_classes"])
    print()
    print_metrics(metrics)
    print()

    print("[5/5] emit")
    path = casting_path(task_dir, spec)
    emit(model, spec, path)
    size_kb = os.path.getsize(path) / 1024
    print(f"      casting: {path}")
    print(f"      size: {size_kb:.1f} KB")
    print()

    print("done")
    print(f"  task dir:      {task_dir}")
    print(f"  train / test:  {len(train_examples)} / {len(test_examples)}")
    print(f"  test accuracy: {metrics['test_accuracy']:.4f}")
    print(f"  casting:      {path} ({size_kb:.1f} KB)")
    print(f"  verify with:   python3 emit.py {task_dir}")
    return task_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--need", required=True, help="plain English description of the task")
    parser.add_argument("--n-train", type=int, default=DEFAULT_N_TRAIN)
    parser.add_argument("--n-test", type=int, default=DEFAULT_N_TEST)
    args = parser.parse_args()
    run(args.need, args.n_train, args.n_test)
