"""Zero-shot Claude baseline, measured against the emitted casting.

Reads a task directory only. Never writes to it, never touches the casting
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

# USD per million tokens for claude-haiku-4-5.
# Source: https://www.anthropic.com/claude/haiku, 8 August 2026, as recorded in
# CLAUDE.md. Never change these from memory; re-verify from that URL.
PRICE_PER_MTOK_INPUT = 1.00
PRICE_PER_MTOK_OUTPUT = 5.00

# Prompt caching and batch multipliers, relative to the base input price.
# Source: https://docs.claude.com/en/docs/build-with-claude/prompt-caching
# (302-redirects to platform.claude.com/docs/en/build-with-claude/prompt-caching),
# verified 2026-08-08: "5-minute cache write tokens are 1.25 times the base
# input tokens price", "Cache read tokens are 0.1 times the base input tokens
# price". Batch API is a 50% discount on input and output, same source.
CACHE_WRITE_MULTIPLIER = 1.25
CACHE_READ_MULTIPLIER = 0.10
BATCH_MULTIPLIER = 0.50

# Minimum cacheable prompt length for claude-haiku-4-5, same source, verified
# 2026-08-08. Below this a cache_control block is ignored with no error and both
# usage cache fields come back 0. Reported, never worked around.
MIN_CACHEABLE_TOKENS = 4096

SYSTEM_TEMPLATE = """You are a text classifier.

Labels:
{labels}

Reply with exactly one label name from the list above and nothing else. No
punctuation, no explanation, no quotes, no preamble."""


def build_system_prompt(spec):
    lines = [f"- {label['name']}: {label['definition']}" for label in spec["labels"]]
    return SYSTEM_TEMPLATE.format(labels="\n".join(lines))


def classify_one(client, system_prompt, text):
    """Return a dict of reply, the four token categories, and seconds.

    Retries at most MAX_RETRIES times on API error. A reply that is not a label
    name is NOT an API error and is never retried.

    The system prompt is identical on every call in a run, so it carries a
    cache_control breakpoint. Whether the cache actually engages is measured,
    not assumed: see MIN_CACHEABLE_TOKENS.
    """
    start = time.perf_counter()
    last_error = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=[{
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }],
                messages=[{"role": "user", "content": text}],
            )
        except anthropic.APIError as exc:
            last_error = exc
            print(f"attempt {attempt + 1}/{MAX_RETRIES + 1} failed: "
                  f"{type(exc).__name__}: {str(exc)[:150]}")
            continue

        reply = "".join(b.text for b in response.content if b.type == "text")
        usage = response.usage
        return {
            "reply": reply,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cache_creation": usage.cache_creation_input_tokens,
            "cache_read": usage.cache_read_input_tokens,
            "seconds": time.perf_counter() - start,
        }

    raise RuntimeError(
        f"call failed after {MAX_RETRIES + 1} attempts: "
        f"{type(last_error).__name__}: {last_error}"
    )


def measure_system_prompt_tokens(spec):
    """Measure the system prompt with count_tokens. Never estimated.

    Counted as (system + probe) minus (probe alone) so the probe message's own
    tokens do not inflate the figure.
    """
    client = anthropic.Anthropic(max_retries=0)
    probe = [{"role": "user", "content": "x"}]
    with_system = client.messages.count_tokens(
        model=MODEL, system=build_system_prompt(spec), messages=probe
    ).input_tokens
    without_system = client.messages.count_tokens(
        model=MODEL, messages=probe
    ).input_tokens
    return {
        "with_system": with_system,
        "without_system": without_system,
        "system_tokens": with_system - without_system,
    }


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
    cache_creation = 0
    cache_read = 0
    cache_write_calls = 0
    cache_read_calls = 0

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        futures = {
            pool.submit(classify_one, client, system_prompt, ex["text"]): i
            for i, ex in enumerate(examples)
        }
        for future in as_completed(futures):
            i = futures[future]
            call = future.result()
            input_tokens += call["input_tokens"]
            output_tokens += call["output_tokens"]
            cache_creation += call["cache_creation"]
            cache_read += call["cache_read"]
            cache_write_calls += 1 if call["cache_creation"] else 0
            cache_read_calls += 1 if call["cache_read"] else 0
            latencies[i] = call["seconds"] * 1000.0
            # Strip transport whitespace only. Anything else that is not an
            # exact label name stays unparseable; it is never coerced.
            stripped = call["reply"].strip()
            replies[i] = stripped
            predictions[i] = name_to_index.get(stripped)

    return {
        "predictions": predictions,
        "replies": replies,
        "latencies": latencies,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_creation": cache_creation,
        "cache_read": cache_read,
        "cache_write_calls": cache_write_calls,
        "cache_read_calls": cache_read_calls,
        "calls": len(examples),
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
    cache_creation = 0
    cache_read = 0
    for ex in sample:
        call = classify_one(client, system_prompt, ex["text"])
        latencies.append(call["seconds"] * 1000.0)
        input_tokens += call["input_tokens"]
        output_tokens += call["output_tokens"]
        cache_creation += call["cache_creation"]
        cache_read += call["cache_read"]

    return {
        "latencies": latencies,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_creation": cache_creation,
        "cache_read": cache_read,
        "n": len(sample),
    }


def run_casting(casting, spec, examples):
    """Classify every example with the emitted casting. Read-only use."""
    name_to_index = {label["name"]: i for i, label in enumerate(spec["labels"])}
    predictions = []
    latencies = []
    for ex in examples:
        start = time.perf_counter()
        label = casting.classify(ex["text"])
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


def cost_usd(input_tokens, output_tokens, cache_creation=0, cache_read=0):
    """Measured cost: each token category at its own multiplier."""
    return (input_tokens * PRICE_PER_MTOK_INPUT
            + cache_creation * PRICE_PER_MTOK_INPUT * CACHE_WRITE_MULTIPLIER
            + cache_read * PRICE_PER_MTOK_INPUT * CACHE_READ_MULTIPLIER
            + output_tokens * PRICE_PER_MTOK_OUTPUT) / 1_000_000


def cost_uncached_usd(total_input_tokens, output_tokens):
    """What the same traffic would cost with every input token at base price."""
    return (total_input_tokens * PRICE_PER_MTOK_INPUT
            + output_tokens * PRICE_PER_MTOK_OUTPUT) / 1_000_000


def corpus_checkpoints(task_dir):
    """Inspect raw_*.jsonl for persisted token usage. Never estimates.

    generate.py writes only the parsed {text, label} items and discards the
    response object, so usage is expected to be absent. This reports what is
    actually on disk rather than assuming either way.
    """
    reports = []
    for name in ("raw_A.jsonl", "raw_B.jsonl"):
        path = os.path.join(task_dir, name)
        if not os.path.exists(path):
            reports.append({"name": name, "present": False})
            continue
        keys = set()
        records = 0
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                keys |= set(json.loads(line).keys())
                records += 1
        reports.append({
            "name": name,
            "present": True,
            "records": records,
            "keys": sorted(keys),
            "has_usage": bool(keys & {"usage", "input_tokens", "output_tokens"}),
        })
    return reports


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


def load_casting(task_dir, spec):
    path = os.path.join(task_dir, f"{spec['task_name']}_classifier.py")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Emit the casting first.")
    module_spec = importlib.util.spec_from_file_location("casting", path)
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


def report(dataset_name, spec, examples, casting, system_tokens):
    labels = [label["name"] for label in spec["labels"]]
    gold = [ex["label"] for ex in examples]

    print(f"=== {dataset_name}: {len(examples)} examples, {spec['num_classes']} classes ===")

    casting_run = run_casting(casting, spec, examples)
    casting_scores = score(gold, casting_run["predictions"], spec["num_classes"])

    baseline_run = run_baseline(spec, examples)
    baseline_scores = score(gold, baseline_run["predictions"], spec["num_classes"])

    # Serial pass runs only after the concurrent pass is fully done, so no
    # requests from it overlap and its timings are contention-free.
    serial_run = time_serial(spec, examples)

    serial_row = "median latency, serial (ms)"
    concurrent_row = f"median latency, under concurrency {CONCURRENCY} (ms)"
    width = max(max(len(name) for name in labels), len(concurrent_row)) + 2
    print()
    print(f"{'metric':<{width}}{'casting':>14}{'haiku zero-shot':>18}")
    print("-" * (width + 32))
    print(f"{'accuracy':<{width}}{casting_scores['accuracy']:>14.4f}"
          f"{baseline_scores['accuracy']:>18.4f}")
    print(f"{'macro F1':<{width}}{casting_scores['macro_f1']:>14.4f}"
          f"{baseline_scores['macro_f1']:>18.4f}")
    print(f"{serial_row:<{width}}"
          f"{statistics.median(casting_run['latencies']):>14.3f}"
          f"{statistics.median(serial_run['latencies']):>18.1f}")
    print(f"{concurrent_row:<{width}}{'-':>14}"
          f"{statistics.median(baseline_run['latencies']):>18.1f}")
    total_input = (baseline_run["input_tokens"] + baseline_run["cache_creation"]
                   + baseline_run["cache_read"])
    print(f"{'uncached input tokens':<{width}}{'-':>14}{baseline_run['input_tokens']:>18d}")
    print(f"{'cache creation tokens':<{width}}{'-':>14}{baseline_run['cache_creation']:>18d}")
    print(f"{'cache read tokens':<{width}}{'-':>14}{baseline_run['cache_read']:>18d}")
    print(f"{'total input tokens':<{width}}{'-':>14}{total_input:>18d}")
    print(f"{'output tokens':<{width}}{'-':>14}{baseline_run['output_tokens']:>18d}")
    print(f"{'unparseable':<{width}}{casting_scores['unparseable']:>14d}"
          f"{baseline_scores['unparseable']:>18d}")

    hit_rate = baseline_run["cache_read"] / total_input if total_input else 0.0
    print()
    print(f"observed cache hit rate: {hit_rate:.2%} "
          f"({baseline_run['cache_read']} cache-read tokens of {total_input} total input)")
    print(f"calls that wrote cache: {baseline_run['cache_write_calls']} of "
          f"{baseline_run['calls']};  calls that read cache: "
          f"{baseline_run['cache_read_calls']} of {baseline_run['calls']}")
    engaged = bool(baseline_run["cache_creation"] or baseline_run["cache_read"])
    print(f"system prompt {system_tokens} tokens vs {MIN_CACHEABLE_TOKENS} minimum "
          f"-> caching {'ENGAGED' if engaged else 'did NOT engage'} "
          f"across all {baseline_run['calls']} calls")
    if engaged:
        print("*** cache tokens are non-zero: caching DID engage. "
              "ACCEPTANCE CHECK FAILED. ***")

    # The casting makes no API calls, so its cost is exactly zero, not rounded.
    uncached = cost_uncached_usd(total_input, baseline_run["output_tokens"])
    batch = uncached * BATCH_MULTIPLIER
    scale = 1_000_000 / len(examples)
    print()
    print(f"{'cost mode':<{width}}{'this run (USD)':>16}{'per 1M (USD)':>16}")
    print("-" * (width + 32))
    print(f"{'uncached (all input at base price)':<{width}}{uncached:>16.6f}"
          f"{uncached * scale:>16.2f}")
    print(f"{'batch (50% of uncached)':<{width}}{batch:>16.6f}{batch * scale:>16.2f}")
    print(f"{'casting':<{width}}{'0':>16}{'0':>16}")
    print(f"caching unavailable, system prompt is {system_tokens} tokens against "
          f"a {MIN_CACHEABLE_TOKENS:,} minimum")
    print()
    print(f"prices: ${PRICE_PER_MTOK_INPUT:.2f}/Mtok in, "
          f"${PRICE_PER_MTOK_OUTPUT:.2f}/Mtok out, cache write x"
          f"{CACHE_WRITE_MULTIPLIER}, cache read x{CACHE_READ_MULTIPLIER} "
          f"(constants at top of file)")
    print(f"serial row: {serial_run['n']} extra calls made one at a time after the "
          f"concurrent pass; predictions discarded,")
    print(f"            {serial_run['input_tokens']} in / {serial_run['output_tokens']} "
          f"out tokens NOT included in the token or cost rows above "
          f"(${cost_usd(serial_run['input_tokens'], serial_run['output_tokens'], serial_run['cache_creation'], serial_run['cache_read']):.6f}).")
    print(f"casting is measured serially in both cases, so it has no "
          f"concurrency-{CONCURRENCY} figure.")
    print()
    print(f"{'per-class recall':<{width}}{'casting':>14}{'haiku zero-shot':>18}")
    print("-" * (width + 32))
    for c, name in enumerate(labels):
        print(f"{name:<{width}}{casting_scores['recall'][c]:>14.4f}"
              f"{baseline_scores['recall'][c]:>18.4f}"
              f"   (support {casting_scores['support'][c]})")

    test = mcnemar_exact(gold, casting_run["predictions"], baseline_run["predictions"])
    print()
    print(f"{'McNemar exact test (paired, two-sided)':<{width}}{'count':>32}")
    print("-" * (width + 32))
    print(f"{'both correct':<{width}}{test['both_correct']:>32d}")
    print(f"{'casting correct, haiku wrong':<{width}}{test['only_a']:>32d}")
    print(f"{'haiku correct, casting wrong':<{width}}{test['only_b']:>32d}")
    print(f"{'both wrong':<{width}}{test['neither']:>32d}")
    print(f"{'discordant pairs':<{width}}{test['discordant']:>32d}")
    print(f"{'two-sided exact p':<{width}}{test['p_value']:>32.4f}")
    if test["discordant"] == 0:
        print("no discordant pairs: the two agree on every example")
    elif test["only_a"] == test["only_b"]:
        print(f"tied: {test['only_a']} discordant pairs each way")
    else:
        winner = "casting" if test["only_a"] > test["only_b"] else "haiku zero-shot"
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
        description="Zero-shot Claude baseline vs the emitted casting."
    )
    parser.add_argument("task_dir", help="directory holding spec.json and test_B.json")
    parser.add_argument("--wild", help="JSON file of {text, label} pairs")
    args = parser.parse_args()

    spec = load_json(os.path.join(args.task_dir, "spec.json"))
    casting = load_casting(args.task_dir, spec)
    print(f"task: {spec['task_name']}   model: {MODEL}   concurrency: {CONCURRENCY}")
    print(f"retries: at most {MAX_RETRIES} per call on API error")

    measured = measure_system_prompt_tokens(spec)
    system_tokens = measured["system_tokens"]
    print(f"system prompt: {system_tokens} tokens, measured via count_tokens "
          f"({measured['with_system']} with system minus "
          f"{measured['without_system']} for the probe message alone)")
    print(f"minimum cacheable prompt for {MODEL}: {MIN_CACHEABLE_TOKENS} tokens")
    print(f"caching can engage: {system_tokens >= MIN_CACHEABLE_TOKENS} "
          f"({system_tokens} vs {MIN_CACHEABLE_TOKENS})")
    print()

    test_examples = normalise_examples(
        load_json(os.path.join(args.task_dir, "test_B.json")), spec, "test_B.json"
    )
    report("test_B", spec, test_examples, casting, system_tokens)

    if args.wild:
        wild_examples = normalise_examples(load_json(args.wild), spec, args.wild)
        report(f"wild ({args.wild})", spec, wild_examples, casting, system_tokens)

    print("=== corpus generation cost ===")
    checkpoints = corpus_checkpoints(args.task_dir)
    for entry in checkpoints:
        if not entry["present"]:
            print(f"{entry['name']}: not present")
        else:
            print(f"{entry['name']}: {entry['records']} records, keys {entry['keys']}")

    if any(entry.get("has_usage") for entry in checkpoints):
        raise NotImplementedError(
            "checkpoints contain usage keys; token totals are recoverable but "
            "this code does not read them yet"
        )
    print("NOT RECOVERABLE: the checkpoints hold only the parsed items. generate.py")
    print("discards the response object, so no usage was ever persisted for the")
    print("corpus generation calls. Corpus generation cost is not estimated here.")


if __name__ == "__main__":
    main()
