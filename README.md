# Foundry

Describe a text classification task in one sentence. Get back a classifier that
runs offline, in about 2 ms, with no dependencies.

Training a small classifier is easy. Getting five thousand labelled examples is
what takes six weeks. Foundry uses Claude to write the label scheme and the
corpus, trains a small network on it, and emits a single self-contained Python
file with the weights embedded. The data was the bottleneck, not the model.

```bash
python3 run.py --need "Triage incoming disaster messages from the public into what emergency responders need to act on."
```

---

## Results

Three tasks, built from the same pipeline with nothing changed but the input
sentence. Every figure below is measured, not estimated.

| | Disaster triage (6 classes) | SMS intent (4 classes) | Supermarket order type (6 classes) |
| --- | --- | --- | --- |
| Casting accuracy, held out | 0.9500 | 1.0000 | 0.9400 |
| Casting macro F1 | 0.9514 | 1.0000 | 0.9261 |
| Haiku 4.5 zero shot, same set | 0.9700 | 0.9800 | not measured |
| Casting latency, median, Apple M4 Max | 2.4 ms | 2.3 ms | 2.6 ms |
| Haiku latency, median serial | 842 ms | 783 ms | not measured |
| Cost per 1M classifications | 0 USD | 0 USD | 0 USD |
| Haiku cost per 1M | 351.95 USD | 329.32 USD | not measured |
| Casting size | 691 KB | 690 KB | 691 KB |
| Training examples | 4000 | 3980 | 3998 |
| Held out examples | 100 | 100 | 100 |

The third task was built through the browser interface with no code changes,
only a different input sentence.

All three tasks requested 4000 training examples. The counts differ because
generate.py deduplicates on normalised text and reports the drop rather than
backfilling. Each casting is compared against Claude on its own held out set, so
the training sizes do not need to match for that comparison to hold.

Latencies are medians of 1000 classifications on an Apple M4 Max and move by a
few tenths of a millisecond between runs, so they are given to one decimal
place. Haiku accuracy moves between 0.97 and 0.98 across repeat runs on the same
set. The casting is deterministic and returns the same answer every time.

### Against its teacher

Pooled across the two tasks with a baseline run, 200 held out examples: the
casting gets 195 correct,
Haiku zero shot gets 195. Discordant pairs split four and four, so McNemar's
exact test gives p = 1.0.

Read that as an absence of evidence for a gap rather than evidence of
equivalence. With only six discordant pairs on the disaster task, the test
cannot reach significance at this sample size even in principle. It rules out a
large difference, not a small one.

### The learning curve

Accuracy against training set size, evaluated on the same fixed held out set,
same seed, so size is the only variable.

| Training examples | Disaster accuracy | SMS accuracy | Supermarket accuracy |
| --- | --- | --- | --- |
| 50 | 0.4700 | 0.9300 | 0.5200 |
| 200 | 0.9000 | 0.9800 | 0.8600 |
| 1000 | 0.9400 | 0.9900 | 0.9100 |
| full | 0.9500 | 1.0000 | 0.9400 |

The full row is the whole deduplicated pool: 4000 disaster, 3980 SMS, 3998
supermarket. A requested size larger than the pool is clamped to the pool and
evaluate.py prints the size it actually used.

Fifty examples is roughly what one person can hand label in an hour.

The curves say different things and all three are worth reporting. On the four
class SMS task with clean separation, fifty examples already gets 93 per cent
and a generated corpus adds little. On the six class disaster task with
overlapping label boundaries, the gap is 48 points, and on the six class
supermarket task it is 42. The value of the data layer scales with how hard the
task is.

---

## Quick start

Requires Python 3.9 or later, `torch`, `numpy` and `anthropic`. Set
`ANTHROPIC_API_KEY` in the environment. Never commit it.

```bash
pip install torch numpy anthropic
export ANTHROPIC_API_KEY=sk-ant-...

# build a model end to end
python3 run.py --need "Classify support tickets by which team should handle them."

# verify the emitted casting matches the trained model
python3 emit.py data/<task_slug>

# compare against Claude zero shot
python3 baseline.py data/<task_slug>

# learning curve
python3 evaluate.py data/<task_slug>
```

### Browser interface

```bash
python3 server.py    # then open http://127.0.0.1:8000
```

`server.py` and `ui.html` must sit in the same directory as `run.py` and
`data/`. The server prints the paths it resolved and the interpreter it will use
for builds, so a misplaced file or a Python without torch is reported at startup
rather than halfway through a build.

Builds run `run.py` as a subprocess, so the pipeline the interface drives is
identical to the command line one. If the Python serving the interface is not
the one with torch installed, set `FOUNDRY_PYTHON` to the correct path.

---

## How it works

```
need (one sentence)
  -> spec.py       Claude writes labels, definitions, edge cases, personas
  -> generate.py   parallel Haiku calls produce the corpus
  -> featurise.py  character n-grams, hashed to 4096 dimensions
  -> train.py      4096 -> 128 ReLU -> classes, Adam, 30 epochs
  -> emit.py       one .py file, weights base64 embedded, stdlib only
```

| File | Role |
| --- | --- |
| `run.py` | orchestrator, the only entry point you need |
| `spec.py` | plain English to a structured label scheme |
| `generate.py` | corpus generation, deduplication, drop accounting |
| `featurise.py` | text to a fixed width vector |
| `train.py` | the model, metrics, and a no-API self test |
| `emit.py` | casting emission and the parity check |
| `evaluate.py` | learning curve |
| `baseline.py` | Claude zero shot comparison, cost, McNemar |
| `validate_wild.py` | checks a hand written evaluation set |
| `server.py`, `ui.html` | local browser interface |

### Design decisions worth knowing

**Feature hashing uses `zlib.crc32`, never Python's built-in `hash`.** String
hashing is randomised per process, so an casting built with `hash` would work
in the session that created it and silently produce different features in a
fresh interpreter. The emitted casting is verified in a subprocess with
`PYTHONHASHSEED` set to a different value, and predictions must match the
trained model exactly.

**Feature, hidden and batch dimensions are 4096, 128 and 64.** All three differ,
so a transposition or stride error cannot hide behind matching shapes.

**Train and test are generated from different scenario seeds.** The seed is
injected into the generation prompt with an instruction that incidents,
locations, names and phrasings must not overlap with any other seed. Measured
exact text overlap between train and test is zero on both tasks.

**Each generation call gets a randomly chosen persona plus randomised length,
formality, typo rate and code mixing.** Without that, several thousand calls
return paraphrases of the same handful of messages.

**First layer weights are int8 with one scale per hidden unit.** The column
scales are applied once after the sparse accumulation rather than once per
feature, which is what keeps latency close to the float32 figure. Applying them
inside the loop cost 3.57 ms against 2.37 ms hoisted on disaster triage, and
3.41 ms against 2.27 ms on SMS intent. Scales are applied before the ReLU, since
ReLU is nonlinear and the ordering is not interchangeable. Measured maximum
quantisation error is 2.3e-03 against a median decision margin of 8.38 on
disaster triage and 10.90 on SMS intent, and the parity check confirms zero
prediction changes on either task.

---

## Evaluation

The held out set is generated by the same model as the training set. That makes
it a weak test on its own, so the repository also carries wild sets: examples
written outside the pipeline that the generator never saw.

| Evaluation set | n | Provenance | Casting accuracy |
| --- | --- | --- | --- |
| `test_B` | 100 | generated, different seed | 0.9500 |
| `wild_disaster.json` | 24 | written by Claude outside the pipeline, not by a human | 0.9167 |
| `wild_human_disaster.json` | 10 | human written and human labelled | 0.6000 |

`wild_disaster.json` shares a model family with the corpus generator, so it is
weaker evidence than the human written set even though it scores higher. At 24
examples one example moves accuracy by 4.2 points.

The human set is the most honest of the three and also the least reliable, at
ten examples. One example moves accuracy by ten points. Two of its four failures
are cases where a human and the model disagreed on which label is correct rather
than clear errors. It covers only four of the six classes, so it cannot measure
the weakest class or detect over triage. Expanding it is the highest value
remaining work.

### The finding that mattered most

The generated held out set saturates at 200 training examples. Wild set accuracy
keeps improving to 4000. Judging on the generated set alone would have suggested
stopping generation at 200 and cost roughly 17 points on real world text.

An evaluation set drawn from the same generator as the training data understates
the value of more data. This is the kind of error a synthetic data pipeline
invites, and it is only visible with an independently written set.

---

## Limitations

**Training data is synthetic.** It reflects what Claude imagines the task looks
like, not a sample of real traffic. Real messages are more fragmented, more
ambiguous and more often unlabellable than anything the generator produces.

**Devanagari input fails.** The code mixing axis generates romanised Hindi only,
so Devanagari script falls outside the training distribution entirely. The
confidence score shows it: 49 per cent on a Devanagari message against 99 per
cent on comparable English. Romanised Hinglish works. Adding a script axis to
the generator is the obvious fix.

**The casting cannot reason.** It is a single feed forward pass. It has no
state, calls no tools, and will never exceed the model that taught it. If Claude
cannot do the task well zero shot, this will not either.

**Prompt caching does not apply.** The obvious objection to the cost comparison
is to cache the fixed system prompt. Claude Haiku 4.5 requires a minimum of 4096
tokens for a cacheable block and these label definitions are about 282 tokens,
so the cache breakpoint is ignored with no error returned. `baseline.py`
measures this rather than assuming it. Batch processing halves the cost but is
asynchronous with up to a 24 hour turnaround, which does not suit real time
classification.

**This is a prioritisation aid, not an autonomous decision system.** In a triage
setting a false negative means a person waits longer for help. The intended
design is a ranked queue that humans still read.

**Nothing here is novel in isolation.** Distillation is long established,
synthetic training data is standard practice, and hashed n-grams into a small
network is a textbook method. What changed is the time and expertise required:
minutes and one sentence instead of weeks and a data pipeline.

---

## Where this fits

Worth using when a task is high volume text classification and the API is not
merely slower but unavailable: no connectivity, data that cannot leave the
device, latency budgets in the low milliseconds, volume in the tens of millions
per day, or a category so new that no labelled data exists yet.

Not worth using for low volume work, anything needing reasoning or generation,
or tasks easy enough that an hour of hand labelling already gets you there. The
SMS learning curve is an example of the last case.

---

## Reproducing every number

```bash
python3 train.py                                        # self test, no API calls
python3 emit.py data/disaster_message_triage            # parity and latency
python3 evaluate.py data/disaster_message_triage        # learning curve
python3 baseline.py data/disaster_message_triage        # Claude comparison, cost, McNemar
python3 validate_wild.py wild_human_disaster.json data/disaster_message_triage
```

`train.py` run on its own trains against a trivially separable synthetic dataset
with no API involvement. If that fails, the pipeline is broken independently of
any data quality question.

Pricing constants in `baseline.py` are 1.00 USD per million input tokens and
5.00 USD per million output tokens for `claude-haiku-4-5`, taken from
https://www.anthropic.com/claude/haiku and verified on 8 August 2026. Re-check
before quoting them.

---

## Licence

MIT.