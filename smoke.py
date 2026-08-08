"""Phase 0 smoke test: torch CPU forward pass + one Anthropic API call."""

import os
import random

import numpy as np
import torch
import anthropic

SEED = 0
FEATURE_DIM = 4096
HIDDEN_DIM = 128
BATCH_SIZE = 64
MODEL = "claude-haiku-4-5-20251001"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def check_torch():
    print("torch version:", torch.__version__)
    print("numpy version:", np.__version__)
    device = torch.device("cpu")
    layer = torch.nn.Linear(FEATURE_DIM, HIDDEN_DIM).to(device)
    x = torch.randn(BATCH_SIZE, FEATURE_DIM, device=device)
    y = layer(x)
    print("input shape:", tuple(x.shape))
    print("output shape:", tuple(y.shape))
    print("output device:", y.device)
    print("output mean:", float(y.detach().mean()))
    assert tuple(y.shape) == (BATCH_SIZE, HIDDEN_DIM)
    print("linear forward pass on CPU: OK")


def check_api():
    print("anthropic version:", anthropic.__version__)
    print("ANTHROPIC_API_KEY set:", "ANTHROPIC_API_KEY" in os.environ)
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=MODEL,
        max_tokens=16,
        messages=[{"role": "user", "content": "Reply with the single word OK."}],
    )
    print("raw response:")
    print(response)


if __name__ == "__main__":
    check_torch()
    print()
    check_api()
