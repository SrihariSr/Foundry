"""Validate a human-written wild set against a task directory.

Exits non-zero if any check fails. Nothing is silently dropped or repaired.
"""

import argparse
import json
import os
import sys

from featurise import NGRAM_SIZES
# Imported rather than reimplemented so the wild set is normalised by exactly
# the same rule the corpus and the featuriser use.
from generate import normalise

MIN_CHARS = min(NGRAM_SIZES)
# 20 characters flagged stock phrases (" breathing properly ") that a human
# writing in the same domain reuses naturally. 40 requires a substantially
# longer verbatim run before it says anything.
SHINGLE = 40
JACCARD_THRESHOLD = 0.6
MIN_RECOMMENDED = 30
MAX_REPORTED = 20


def load_json(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    with open(path) as f:
        return json.load(f)


def corpus_texts(task_dir):
    """All training and test texts, tagged with their source file."""
    texts = []
    for name in ("train_A.json", "test_B.json"):
        for i, entry in enumerate(load_json(os.path.join(task_dir, name))):
            texts.append((f"{name}[{i}]", entry["text"]))
    return texts


def build_shingles(texts, size=SHINGLE):
    shingles = set()
    for _, text in texts:
        normalised = normalise(text)
        for i in range(len(normalised) - size + 1):
            shingles.add(normalised[i:i + size])
    return shingles


def jaccard(tokens_a, tokens_b):
    union = tokens_a | tokens_b
    if not union:
        return 0.0
    return len(tokens_a & tokens_b) / len(union)


def closest_by_jaccard(tokens, corpus_tokens):
    """Return (score, source, text) for the corpus entry with the highest overlap."""
    best_score = 0.0
    best = (None, None)
    for source, text, other in corpus_tokens:
        score = jaccard(tokens, other)
        if score > best_score:
            best_score = score
            best = (source, text)
    return best_score, best[0], best[1]


def find_source(texts, fragment):
    for source, text in texts:
        if fragment in normalise(text):
            return source, text
    return None, None


def main(wild_path, task_dir):
    spec = load_json(os.path.join(task_dir, "spec.json"))
    labels = [label["name"] for label in spec["labels"]]
    wild = load_json(wild_path)
    if not isinstance(wild, list):
        raise ValueError(f"{wild_path} must be a JSON array, got {type(wild).__name__}")

    print(f"wild set: {wild_path}")
    print(f"task:     {task_dir}  ({spec['task_name']}, {len(labels)} labels)")
    print(f"entries:  {len(wild)}")
    print()

    failures = []
    warnings = []

    # --- structure and labels ---------------------------------------------
    bad_labels = []
    for i, entry in enumerate(wild):
        if not isinstance(entry, dict):
            failures.append(f"entry {i} is {type(entry).__name__}, not an object")
            continue
        if "text" not in entry or not isinstance(entry["text"], str):
            failures.append(f"entry {i} has no string 'text' field")
            continue
        if "label" not in entry:
            failures.append(f"entry {i} has no 'label' field")
            continue
        if entry["label"] not in labels:
            bad_labels.append((i, entry["label"]))

    print("--- labels ---")
    if bad_labels:
        for i, label in bad_labels[:MAX_REPORTED]:
            print(f"  INVALID entry {i}: {label!r}")
        if len(bad_labels) > MAX_REPORTED:
            print(f"  ... and {len(bad_labels) - MAX_REPORTED} more")
        failures.append(f"{len(bad_labels)} entries carry a label not in spec.json")
    else:
        print("  all labels are exact spec label names")

    # --- class balance -----------------------------------------------------
    counts = {name: 0 for name in labels}
    for entry in wild:
        if isinstance(entry, dict) and entry.get("label") in counts:
            counts[entry["label"]] += 1
    print()
    print("--- class balance ---")
    for name in labels:
        print(f"  {name:<32}{counts[name]:>5}")
    empty = [name for name in labels if counts[name] == 0]
    if empty:
        print(f"  WARNING: {len(empty)} label(s) with zero examples: {', '.join(empty)}")
        warnings.append(f"{len(empty)} label(s) have zero examples")

    # --- too short ---------------------------------------------------------
    print()
    print(f"--- shorter than {MIN_CHARS} chars after whitespace collapsing ---")
    short = [
        (i, entry["text"]) for i, entry in enumerate(wild)
        if isinstance(entry, dict) and isinstance(entry.get("text"), str)
        and len(normalise(entry["text"])) < MIN_CHARS
    ]
    if short:
        for i, text in short[:MAX_REPORTED]:
            print(f"  entry {i}: {text!r} -> {normalise(text)!r}")
        failures.append(f"{len(short)} entries are too short for featurise()")
    else:
        print("  none")

    # --- exact collisions --------------------------------------------------
    corpus = corpus_texts(task_dir)
    by_normalised = {}
    for source, text in corpus:
        by_normalised.setdefault(normalise(text), source)

    print()
    print(f"--- exact collisions against train_A.json + test_B.json "
          f"({len(corpus)} corpus entries) ---")
    collisions = []
    for i, entry in enumerate(wild):
        if not isinstance(entry, dict) or not isinstance(entry.get("text"), str):
            continue
        source = by_normalised.get(normalise(entry["text"]))
        if source:
            collisions.append((i, source, entry["text"]))
    if collisions:
        for i, source, text in collisions[:MAX_REPORTED]:
            print(f"  entry {i} matches {source}: {text!r}")
        if len(collisions) > MAX_REPORTED:
            print(f"  ... and {len(collisions) - MAX_REPORTED} more")
        failures.append(f"{len(collisions)} entries exactly match a corpus entry")
    else:
        print("  none")

    # --- near duplicates ---------------------------------------------------
    colliding = {i for i, _, _ in collisions}
    shingles = build_shingles(corpus)
    print()
    print(f"--- near duplicates ({SHINGLE}-char shared substring, "
          f"{len(shingles)} corpus shingles) ---")
    near = []
    for i, entry in enumerate(wild):
        if i in colliding or not isinstance(entry, dict):
            continue
        if not isinstance(entry.get("text"), str):
            continue
        normalised = normalise(entry["text"])
        hit = next(
            (normalised[j:j + SHINGLE] for j in range(len(normalised) - SHINGLE + 1)
             if normalised[j:j + SHINGLE] in shingles),
            None,
        )
        if hit:
            source, text = find_source(corpus, hit)
            near.append((i, hit, source, text))
    if near:
        for i, hit, source, text in near[:MAX_REPORTED]:
            print(f"  entry {i} shares {hit!r} with {source}: {text[:70]!r}")
        if len(near) > MAX_REPORTED:
            print(f"  ... and {len(near) - MAX_REPORTED} more")
        # Advisory only: a human writing about the same domain reuses phrasing.
        warnings.append(
            f"{len(near)} entries share a {SHINGLE}-char substring with the corpus"
        )
    else:
        print("  none")
    if colliding:
        print(f"  ({len(colliding)} exact collisions excluded, already reported above)")

    # --- token-level Jaccard ------------------------------------------------
    corpus_tokens = [(source, text, set(normalise(text).split())) for source, text in corpus]
    print()
    print(f"--- token Jaccard vs closest corpus entry (flagged above "
          f"{JACCARD_THRESHOLD}) ---")
    flagged = []
    for i, entry in enumerate(wild):
        if not isinstance(entry, dict) or not isinstance(entry.get("text"), str):
            continue
        tokens = set(normalise(entry["text"]).split())
        score, source, text = closest_by_jaccard(tokens, corpus_tokens)
        mark = "  FLAG" if score > JACCARD_THRESHOLD else ""
        print(f"  entry {i:>3}  {score:.3f}  {source or '(no token overlap)'}{mark}")
        if score > JACCARD_THRESHOLD:
            flagged.append((i, score, source, text))
    if flagged:
        for i, score, source, text in flagged:
            print(f"  entry {i} at {score:.3f} vs {source}: {text[:70]!r}")
        warnings.append(
            f"{len(flagged)} entries exceed {JACCARD_THRESHOLD} token Jaccard "
            f"against a corpus entry"
        )
    elif wild:
        print(f"  none above {JACCARD_THRESHOLD}")

    # --- size --------------------------------------------------------------
    print()
    print("--- size ---")
    print(f"  {len(wild)} entries")
    if len(wild) < MIN_RECOMMENDED:
        if wild:
            print(f"  WARNING: under {MIN_RECOMMENDED}. One example moves accuracy by "
                  f"{100 / len(wild):.1f} points, so the figure is not stable.")
        else:
            print("  WARNING: the set is empty; nothing can be measured from it.")
        warnings.append(f"only {len(wild)} entries, under {MIN_RECOMMENDED}")

    # --- verdict -----------------------------------------------------------
    print()
    for warning in warnings:
        print(f"WARNING: {warning}")
    for failure in failures:
        print(f"FAILURE: {failure}")
    print()
    if failures:
        print(f"FAIL ({len(failures)} failure(s), {len(warnings)} warning(s))")
        return 1
    print(f"PASS ({len(warnings)} warning(s))")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate a wild set against a task.")
    parser.add_argument("wild_file", help="JSON array of {text, label[, notes]}")
    parser.add_argument("task_dir", help="directory holding spec.json, train_A.json, test_B.json")
    args = parser.parse_args()
    sys.exit(main(args.wild_file, args.task_dir))
