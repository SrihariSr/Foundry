# Foundry: rules of engagement

## What we are building
A pipeline that turns a plain-English classification task into a small
deployable model. Claude generates the labelled corpus, PyTorch trains a
small MLP, output is a single self-contained Python file.

This is a 5-hour hackathon prototype. Speed of validation beats quality.

## Hard rules

1. ONE module per turn. Write it, run it, paste real output, then STOP and
   wait. Do not chain ahead to the next phase.
2. Never claim something works without executing it. Paste actual stdout.
   No predicted or illustrative output.
3. No try/except that swallows errors. No bare except. Fail loudly with the
   real traceback.
4. No mock data, no placeholder returns, no "TODO fill in later" stubs in
   any code path that the pipeline actually runs.
5. No fallback paths that silently substitute defaults when something fails.
   If an API call fails, crash.
6. Approved dependencies ONLY: torch, anthropic, numpy. Nothing else without
   asking first. No sklearn, no transformers, no pandas, no rich, no typer.
7. Do not refactor code that already passes its acceptance check.
8. Do not add CLI polish, progress bars, colour output, logging frameworks,
   config files, or a web UI. Plain print() is correct.
9. If an acceptance check fails, STOP and report the actual numbers. Do not
   work around it, do not loosen the check, do not retry with different
   settings unless told.
10. Never edit a test or acceptance threshold to make it pass.

## Fixed technical decisions (do not change these)

- Feature dim: 4096. Hidden dim: 128. Batch size: 64.
  These are deliberately all different so shape bugs cannot hide.
- Hashing: zlib.crc32(ngram.encode("utf-8")) % 4096.
  NEVER use Python's builtin hash() on strings. It is randomised per
  process and will silently break the emitted artifact.
- Features: character n-grams n=3,4,5, lowercased, whitespace collapsed,
  then L2 normalise the vector.
- Model: Linear(4096,128) -> ReLU -> Linear(128,num_classes).
  CrossEntropyLoss, Adam lr=1e-3, 30 epochs.
- Device: CPU. The model is tiny; MPS transfer overhead is not worth it.
- Seed everything (random, numpy, torch) with seed=0 at the top of every
  script that has randomness.
- Pricing constants are $1.00/$5.00 per MTok for claude-haiku-4-5, sourced
  from https://www.anthropic.com/claude/haiku on 8 August 2026. Never change
  these from memory. If pricing is questioned, re-verify from that URL.
- Always report standard, batch and cached costs together. Never quote the
  standard figure alone.
- Haiku 4.5 minimum cacheable prompt is 4,096 tokens (docs.claude.com,
  verified 8 Aug 2026). Our system prompt is ~282 tokens, so caching cannot
  engage. Never claim a cached cost figure for this workload.

## File layout
foundry/
  featurise.py   # text -> vector
  train.py       # dataset -> model + metrics
  generate.py    # spec -> labelled examples via Claude
  spec.py        # plain English -> task spec JSON
  evaluate.py    # metrics, confusion matrix, learning curve
  emit.py        # model -> single-file artifact
  run.py         # orchestrator
  data/          # cached JSON, gitignored

## Reporting format after every module
- What you ran (exact command)
- Real stdout
- Whether the acceptance check passed, with the actual numbers
- Anything you had to guess or assume
