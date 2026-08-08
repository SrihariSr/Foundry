"""Spec -> labelled corpus via parallel Claude calls."""

import json
import os
import random
import threading
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed

import anthropic

from featurise import NGRAM_SIZES
from spec import parse_json

MODEL = "claude-haiku-4-5-20251001"
CONCURRENCY = 20
PER_CALL = 20
MIN_TEXT_CHARS = min(NGRAM_SIZES)
DATA_DIR = "data"

LENGTHS = [
    "a handful of words, terse, like a rushed SMS",
    "one short sentence",
    "two or three sentences",
    "a long rambling paragraph with several separate points",
]
FORMALITIES = [
    "formal and official, like a report filed by an institution",
    "neutral and plain",
    "casual and conversational",
    "very informal, slang-heavy, shouty, lots of caps or punctuation",
]
TYPO_RATES = [
    "clean spelling and grammar",
    "occasional typos and missing punctuation",
    "frequent typos, phonetic spellings, missing vowels, autocorrect mangling",
]
CODE_MIXING = [
    "plain English only",
    "English with a few non-English words or place names left untranslated",
    "English heavily mixed with transliterated words from another language",
]

SYSTEM_PROMPT = """You write realistic training data for a text classifier.

Return ONLY a raw JSON array. No markdown fences, no preamble, no commentary.
Your entire response must parse with json.loads().

Each element must be exactly: {"text": "<the message>", "label": "<label name>"}

The label must be one of the label names given to you, spelled exactly.
Write the messages as they would really be written by the person described,
not as clean textbook examples. Do not number them, do not add meta-commentary,
do not mention the persona or the style instructions in the text itself."""


def normalise(text: str) -> str:
    return " ".join(text.lower().split())


def build_prompt(spec, persona, style, seed_tag, count):
    label_lines = []
    for label in spec["labels"]:
        edges = "; ".join(str(e) for e in label["examples_of_edge_cases"])
        label_lines.append(
            f"- {label['name']}: {label['definition']}\n  hard cases: {edges}"
        )
    labels_block = "\n".join(label_lines)

    return f"""Task: {spec['task_name']}

Labels:
{labels_block}

Write {count} messages in English.

Writer persona for every message in this batch:
{persona}

Style constraints for this batch:
- Length: {style['length']}
- Register: {style['formality']}
- Spelling: {style['typos']}
- Language mixing: {style['mixing']}

Scenario seed: {seed_tag}
Treat this seed as a distinct scenario universe. Incidents, locations, names,
and phrasings in this batch must not overlap with what you would write for a
different seed.

Spread the {count} messages across all {spec['num_classes']} labels. Include
several genuinely hard or ambiguous cases that sit near a boundary between two
labels. Every message must be distinct in content, not a reworded twin of
another."""


def call_once(client, spec, persona, style, seed_tag, count):
    response = client.messages.create(
        model=MODEL,
        max_tokens=8000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_prompt(spec, persona, style, seed_tag, count)}],
    )
    if response.stop_reason != "end_turn":
        print(f"warning: stop_reason={response.stop_reason!r} "
              f"(output likely truncated, JSON may not parse)")
    text = "".join(block.text for block in response.content if block.type == "text")
    items = parse_json(text)
    if not isinstance(items, list):
        raise ValueError(f"expected a JSON array, got {type(items).__name__}")
    return items


def generate(spec, n_examples, seed_tag, out_dir=DATA_DIR):
    """Return a deduplicated list of {"text": str, "label": int}."""
    client = anthropic.Anthropic()
    name_to_index = {label["name"]: i for i, label in enumerate(spec["labels"])}
    personas = spec["generation_personas"]

    # crc32, not builtin hash(): hash() is randomised per process.
    rng = random.Random(zlib.crc32(seed_tag.encode("utf-8")))

    n_calls = -(-n_examples // PER_CALL)
    jobs = []
    for _ in range(n_calls):
        jobs.append(
            (
                rng.choice(personas),
                {
                    "length": rng.choice(LENGTHS),
                    "formality": rng.choice(FORMALITIES),
                    "typos": rng.choice(TYPO_RATES),
                    "mixing": rng.choice(CODE_MIXING),
                },
            )
        )

    # Each batch is appended to disk as it lands, so a late failure loses nothing.
    os.makedirs(out_dir, exist_ok=True)
    checkpoint = os.path.join(out_dir, f"raw_{seed_tag}.jsonl")
    open(checkpoint, "w").close()
    lock = threading.Lock()

    def run_batch(persona, style):
        items = call_once(client, spec, persona, style, seed_tag, PER_CALL)
        with lock:
            with open(checkpoint, "a") as f:
                for item in items:
                    f.write(json.dumps(item) + "\n")
        return items

    raw_items = []
    failures = []
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        futures = {
            pool.submit(run_batch, persona, style): i
            for i, (persona, style) in enumerate(jobs)
        }
        for future in as_completed(futures):
            i = futures[future]
            try:
                raw_items.extend(future.result())
            except Exception as exc:
                # Reported, never swallowed; and if every batch fails we crash below.
                failures.append((i, type(exc).__name__, str(exc)[:300]))
                print(f"batch {i} FAILED: {type(exc).__name__}: {str(exc)[:300]}")

    if not raw_items:
        raise RuntimeError(
            f"all {n_calls} batches failed for seed_tag={seed_tag!r}; "
            f"first failure: {failures[0] if failures else 'none recorded'}"
        )

    examples = []
    seen = set()
    drops = {
        "not_an_object": 0,
        "missing_or_bad_text": 0,
        "missing_label": 0,
        "unknown_label": 0,
        "text_too_short": 0,
        "duplicate": 0,
    }
    dropped_samples = []

    def drop(reason, item):
        drops[reason] += 1
        if reason != "duplicate" and len(dropped_samples) < 5:
            dropped_samples.append((reason, item))

    for item in raw_items:
        if not isinstance(item, dict):
            drop("not_an_object", item)
            continue
        if not isinstance(item.get("text"), str):
            drop("missing_or_bad_text", item)
            continue
        if "label" not in item:
            drop("missing_label", item)
            continue
        if item["label"] not in name_to_index:
            drop("unknown_label", item)
            continue
        key = normalise(item["text"])
        if len(key) < MIN_TEXT_CHARS:
            # featurise() cannot build an n-gram from this and would crash later.
            drop("text_too_short", item)
            continue
        if key in seen:
            drop("duplicate", item)
            continue
        seen.add(key)
        examples.append({"text": item["text"], "label": name_to_index[item["label"]]})

    balance = {}
    for ex in examples:
        name = spec["labels"][ex["label"]]["name"]
        balance[name] = balance.get(name, 0) + 1

    malformed = sum(count for reason, count in drops.items() if reason != "duplicate")
    tag = f"[seed_tag={seed_tag}]"
    print(f"{tag} requested: {n_examples} ({n_calls} calls x {PER_CALL})")
    print(f"{tag} failed batches: {len(failures)} of {n_calls}")
    print(f"{tag} returned:  {len(raw_items)}")
    print(f"{tag} dropped malformed: {malformed} "
          f"{ {r: c for r, c in drops.items() if r != 'duplicate' and c} }")
    print(f"{tag} dropped duplicates: {drops['duplicate']}")
    print(f"{tag} after dedup: {len(examples)} "
          f"({len(examples) / n_examples:.1%} of requested)")
    print(f"{tag} class balance: "
          f"{dict(sorted(balance.items(), key=lambda kv: -kv[1]))}")
    for reason, item in dropped_samples:
        print(f"{tag} dropped sample [{reason}]: {json.dumps(item)[:200]}")
    print(f"{tag} raw batches checkpointed to {checkpoint}")
    return examples
