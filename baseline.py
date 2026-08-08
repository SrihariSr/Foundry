"""Zero-shot Claude baseline, measured against the emitted artifact.

Reads a task directory only. Never writes to it, never touches the artifact
beyond importing it and calling classify().
"""

import argparse
import importlib.util
import json
import math
import os
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import anthropic

MODEL = "claude-haiku-4-5-20251001"
CONCURRENCY = 20
MAX_RETRIES = 2  # retries after the first attempt, so 3 attempts at most
MAX_TOKENS = 64
SERIAL_TIMING_N = 10
SEED = 0

SYSTEM_TEMPLATE = """You are a text classifier.

Labels:
{labels}

Reply with exactly one label name from the list above and nothing else. No
punctuation, no explanation, no quotes, no preamble."""


def build_system_prompt(spec):
    lines = [f"- {label['name']}: {label['definition']}" for label in spec["labels"]]
    return SYSTEM_TEMPLATE.format(labels="\n".join(lines))


def classify_one(client, system_prompt, text):
    """Return (reply_text, input_tokens, output_tokens, seconds).

    Retries at most MAX_RETRIES times on API error. A reply that is not a label
    name is NOT an API error and is never retried.
    """
    start = time.perf_counter()
    last_error = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=system_prompt,
                messages=[{"role": "user", "content": text}],
            )
        except anthropic.APIError as exc:
            last_error = exc
            print(f"attempt {attempt + 1}/{MAX_RETRIES + 1} failed: "
                  f"{type(exc).__name__}: {str(exc)[:150]}")
            continue

        reply = "".join(b.text for b in response.content if b.type == "text")
        elapsed = time.perf_counter() - start
        return reply, response.usage.input_tokens, response.usage.output_tokens, elapsed

    raise RuntimeError(
        f"call failed after {MAX_RETRIES + 1} attempts: "
        f"{type(last_error).__name__}: {last_error}"
    )


def run_baseline(spec, examples):
    """Classify every example zero-shot. Returns predictions and call stats."""
    client = anthropic.Anthropic(max_retries=0)  # retries are ours, not the SDK's
    system_prompt = build_system_prompt(spec)
    name_to_index = {label["name"]: i for i, label in enumerate(spec["labels"])}

    predictions = [None] * len(examples)
    replies = [None] * len(examples)
    latencies = [0.0] * len(examples)
    input_tokens = 0
    output_tokens = 0

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        futures = {
            pool.submit(classify_one, client, system_prompt, ex["text"]): i
            for i, ex in enumerate(examples)
        }
        for future in as_completed(futures):
            i = futures[future]
            reply, in_tok, out_tok, elapsed = future.result()
            input_tokens += in_tok
            output_tokens += out_tok
            latencies[i] = elapsed * 1000.0
            # Strip transport whitespace only. Anything else that is not an
            # exact label name stays unparseable; it is never coerced.
            stripped = reply.strip()
            replies[i] = stripped
            predictions[i] = name_to_index.get(stripped)

    return {
        "predictions": predictions,
        "replies": replies,
        "latencies": latencies,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


def time_serial(spec, examples, n=SERIAL_TIMING_N):
    """Time n classifications one at a time, no threading.

    The concurrent pass measures per-call wall time while 20 requests are in
    flight, so it absorbs thread contention. This pass isolates true per-call
    latency. Predictions are discarded; only timing is used.
    """
    client = anthropic.Anthropic(max_retries=0)
    system_prompt = build_system_prompt(spec)
    chooser = random.Random(SEED)
    sample = chooser.sample(examples, min(n, len(examples)))

    latencies = []
    input_tokens = 0
    output_tokens = 0
    for ex in sample:
        _, in_tok, out_tok, elapsed = classify_one(client, system_prompt, ex["text"])
        latencies.append(elapsed * 1000.0)
        input_tokens += in_tok
        output_tokens += out_tok

    return {
        "latencies": latencies,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "n": len(sample),
    }


def run_artifact(artifact, spec, examples):
    """Classify every example with the emitted artifact. Read-only use."""
    name_to_index = {label["name"]: i for i, label in enumerate(spec["labels"])}
    predictions = []
    latencies = []
    for ex in examples:
        start = time.perf_counter()
        label = artifact.classify(ex["text"])
        latencies.append((time.perf_counter() - start) * 1000.0)
        predictions.append(name_to_index[label])
    return {"predictions": predictions, "latencies": latencies}


def score(gold, predictions, num_classes):
    """Metrics where an unparseable prediction (None) counts against recall."""
    matrix = [[0] * num_classes for _ in range(num_classes)]
    unparseable_by_class = [0] * num_classes
    for true_index, predicted in zip(gold, predictions):
        if predicted is None:
            unparseable_by_class[true_index] += 1
        else:
            matrix[true_index][predicted] += 1

    precision, recall, f1, support = [], [], [], []
    for c in range(num_classes):
        correct = matrix[c][c]
        predicted_c = sum(matrix[r][c] for r in range(num_classes))
        actual_c = sum(matrix[c]) + unparseable_by_class[c]
        p = correct / predicted_c if predicted_c else 0.0
        r = correct / actual_c if actual_c else 0.0
        precision.append(p)
        recall.append(r)
        f1.append(2 * p * r / (p + r) if (p + r) else 0.0)
        support.append(actual_c)

    correct_total = sum(matrix[c][c] for c in range(num_classes))
    unparseable = sum(unparseable_by_class)
    parseable = len(gold) - unparseable
    return {
        "accuracy": correct_total / len(gold),
        "accuracy_parseable_only": correct_total / parseable if parseable else 0.0,
        "macro_f1": sum(f1) / num_classes,
        "precision": precision,
        "recall": recall,
        "support": support,
        "unparseable": unparseable,
        "unparseable_by_class": unparseable_by_class,
        "confusion_matrix": matrix,
    }


def mcnemar_exact(gold, predictions_a, predictions_b):
    """Two-sided exact McNemar test on paired predictions.

    Only the discordant pairs carry information: b = a correct while b wrong,
    c = b correct while a wrong. Under the null the two classifiers are equally
    likely to win a discordant pair, so b ~ Binomial(b + c, 0.5). The exact
    two-sided p-value doubles the smaller tail. An unparseable prediction
    (None) is incorrect, never a match.
    """
    both_correct = only_a = only_b = neither = 0
    for true_label, pred_a, pred_b in zip(gold, predictions_a, predictions_b):
        a_correct = pred_a == true_label
        b_correct = pred_b == true_label
        if a_correct and b_correct:
            both_correct += 1
        elif a_correct:
            only_a += 1
        elif b_correct:
            only_b += 1
        else:
            neither += 1

    discordant = only_a + only_b
    if discordant == 0:
        p_value = 1.0
    else:
        smaller = min(only_a, only_b)
        tail = sum(math.comb(discordant, k) for k in range(smaller + 1)) / (2 ** discordant)
        p_value = min(1.0, 2.0 * tail)

    return {
        "both_correct": both_correct,
        "only_a": only_a,
        "only_b": only_b,
        "neither": neither,
        "discordant": discordant,
        "p_value": p_value,
    }


def load_json(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    with open(path) as f:
        return json.load(f)


def load_artifact(task_dir, spec):
    path = os.path.join(task_dir, f"{spec['task_name']}_classifier.py")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Emit the artifact first.")
    module_spec = importlib.util.spec_from_file_location("artifact", path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def normalise_examples(raw, spec, source):
    """Accept {text, label} with label as an index or an exact label name."""
    name_to_index = {label["name"]: i for i, label in enumerate(spec["labels"])}
    examples = []
    for i, item in enumerate(raw):
        if "text" not in item or "label" not in item:
            raise ValueError(f"{source}[{i}] needs both 'text' and 'label': {item}")
        label = item["label"]
        if isinstance(label, bool) or not isinstance(label, (int, str)):
            raise ValueError(f"{source}[{i}] label must be an int or str: {label!r}")
        if isinstance(label, int):
            if not 0 <= label < spec["num_classes"]:
                raise ValueError(f"{source}[{i}] label index {label} out of range")
            index = label
        else:
            if label not in name_to_index:
                raise ValueError(
                    f"{source}[{i}] label {label!r} is not in the spec; "
                    f"valid labels are {sorted(name_to_index)}"
                )
            index = name_to_index[label]
        examples.append({"text": item["text"], "label": index})
    return examples


def report(dataset_name, spec, examples, artifact):
    labels = [label["name"] for label in spec["labels"]]
    gold = [ex["label"] for ex in examples]

    print(f"=== {dataset_name}: {len(examples)} examples, {spec['num_classes']} classes ===")

    artifact_run = run_artifact(artifact, spec, examples)
    artifact_scores = score(gold, artifact_run["predictions"], spec["num_classes"])

    baseline_run = run_baseline(spec, examples)
    baseline_scores = score(gold, baseline_run["predictions"], spec["num_classes"])

    # Serial pass runs only after the concurrent pass is fully done, so no
    # requests from it overlap and its timings are contention-free.
    serial_run = time_serial(spec, examples)

    serial_row = "median latency, serial (ms)"
    concurrent_row = f"median latency, under concurrency {CONCURRENCY} (ms)"
    width = max(max(len(name) for name in labels), len(concurrent_row)) + 2
    print()
    print(f"{'metric':<{width}}{'artifact':>14}{'haiku zero-shot':>18}")
    print("-" * (width + 32))
    print(f"{'accuracy':<{width}}{artifact_scores['accuracy']:>14.4f}"
          f"{baseline_scores['accuracy']:>18.4f}")
    print(f"{'macro F1':<{width}}{artifact_scores['macro_f1']:>14.4f}"
          f"{baseline_scores['macro_f1']:>18.4f}")
    print(f"{serial_row:<{width}}"
          f"{statistics.median(artifact_run['latencies']):>14.3f}"
          f"{statistics.median(serial_run['latencies']):>18.1f}")
    print(f"{concurrent_row:<{width}}{'-':>14}"
          f"{statistics.median(baseline_run['latencies']):>18.1f}")
    print(f"{'input tokens':<{width}}{'-':>14}{baseline_run['input_tokens']:>18d}")
    print(f"{'output tokens':<{width}}{'-':>14}{baseline_run['output_tokens']:>18d}")
    print(f"{'unparseable':<{width}}{artifact_scores['unparseable']:>14d}"
          f"{baseline_scores['unparseable']:>18d}")
    print()
    print(f"serial row: {serial_run['n']} extra calls made one at a time after the "
          f"concurrent pass; predictions discarded,")
    print(f"            {serial_run['input_tokens']} in / {serial_run['output_tokens']} "
          f"out tokens NOT included in the token rows above.")
    print(f"artifact is measured serially in both cases, so it has no "
          f"concurrency-{CONCURRENCY} figure.")
    print()
    print(f"{'per-class recall':<{width}}{'artifact':>14}{'haiku zero-shot':>18}")
    print("-" * (width + 32))
    for c, name in enumerate(labels):
        print(f"{name:<{width}}{artifact_scores['recall'][c]:>14.4f}"
              f"{baseline_scores['recall'][c]:>18.4f}"
              f"   (support {artifact_scores['support'][c]})")

    test = mcnemar_exact(gold, artifact_run["predictions"], baseline_run["predictions"])
    print()
    print(f"{'McNemar exact test (paired, two-sided)':<{width}}{'count':>32}")
    print("-" * (width + 32))
    print(f"{'both correct':<{width}}{test['both_correct']:>32d}")
    print(f"{'artifact correct, haiku wrong':<{width}}{test['only_a']:>32d}")
    print(f"{'haiku correct, artifact wrong':<{width}}{test['only_b']:>32d}")
    print(f"{'both wrong':<{width}}{test['neither']:>32d}")
    print(f"{'discordant pairs':<{width}}{test['discordant']:>32d}")
    print(f"{'two-sided exact p':<{width}}{test['p_value']:>32.4f}")
    if test["discordant"] == 0:
        print("no discordant pairs: the two agree on every example")
    elif test["only_a"] == test["only_b"]:
        print(f"tied: {test['only_a']} discordant pairs each way")
    else:
        winner = "artifact" if test["only_a"] > test["only_b"] else "haiku zero-shot"
        print(f"{winner} wins {max(test['only_a'], test['only_b'])} of "
              f"{test['discordant']} discordant pairs")

    if baseline_scores["unparseable"]:
        print()
        print(f"unparseable replies ({baseline_scores['unparseable']}), "
              f"accuracy over parseable only: "
              f"{baseline_scores['accuracy_parseable_only']:.4f}")
        shown = 0
        for i, predicted in enumerate(baseline_run["predictions"]):
            if predicted is None and shown < 5:
                print(f"  reply {baseline_run['replies'][i]!r} "
                      f"for text {examples[i]['text'][:60]!r}")
                shown += 1
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Zero-shot Claude baseline vs the emitted artifact."
    )
    parser.add_argument("task_dir", help="directory holding spec.json and test_B.json")
    parser.add_argument("--wild", help="JSON file of {text, label} pairs")
    args = parser.parse_args()

    spec = load_json(os.path.join(args.task_dir, "spec.json"))
    artifact = load_artifact(args.task_dir, spec)
    print(f"task: {spec['task_name']}   model: {MODEL}   concurrency: {CONCURRENCY}")
    print(f"retries: at most {MAX_RETRIES} per call on API error")
    print()

    test_examples = normalise_examples(
        load_json(os.path.join(args.task_dir, "test_B.json")), spec, "test_B.json"
    )
    report("test_B", spec, test_examples, artifact)

    if args.wild:
        wild_examples = normalise_examples(load_json(args.wild), spec, args.wild)
        report(f"wild ({args.wild})", spec, wild_examples, artifact)


if __name__ == "__main__":
    main()
