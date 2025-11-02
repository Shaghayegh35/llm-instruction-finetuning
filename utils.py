
"""
Utility helpers for instruction-style SFT datasets and reproducible runs.

This module focuses on:
- Robust prompt formatting (Instruction / Input / Response)
- Light validation of dataset records
- Reusable helpers: seeding, safe truncation, token counting

Author: Shaghayegh Khalighiyan
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Literal, Any
import random

# -----------------------------
# Reproducibility helpers
# -----------------------------

def set_global_seed(seed: int = 42) -> None:
    """Set Python/random and common libs to a fixed seed if available."""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


# -----------------------------
# Prompt formatting
# -----------------------------

DEFAULT_TEMPLATE = """### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

NO_INPUT_TEMPLATE = """### Instruction:
{instruction}

### Response:
{output}"""

@dataclass(frozen=True)
class Record:
    """Typed view over a JSONL row for SFT."""
    instruction: str
    output: str
    input: Optional[str] = None

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Record":
        inst = (d.get("instruction") or "").strip()
        out = (d.get("output") or "").strip()
        inp = d.get("input")
        inp = None if inp is None else str(inp).strip()
        if not inst:
            raise ValueError("record is missing non-empty 'instruction'")
        if not out:
            raise ValueError("record is missing non-empty 'output'")
        return Record(instruction=inst, output=out, input=inp)


def format_example(
    rec: Dict[str, Any] | Record,
    *,
    style: Literal["default", "no_input_auto"] = "no_input_auto",
    template: Optional[str] = None,
) -> str:
    """Format a dataset record into an instruction-style prompt."""
    record = rec if isinstance(rec, Record) else Record.from_dict(rec)
    if style == "no_input_auto":
        chosen = NO_INPUT_TEMPLATE if not record.input else DEFAULT_TEMPLATE
    else:
        chosen = DEFAULT_TEMPLATE
    if template is not None:
        chosen = template
    return chosen.format(
        instruction=record.instruction,
        input=(record.input or ""),
        output=record.output,
    )


# -----------------------------
# Utility helpers
# -----------------------------

def safe_truncate(text: str, max_len: int) -> str:
    """Truncate without breaking mid-string when max_len <= 0, return empty."""
    if max_len <= 0:
        return ""
    return text if len(text) <= max_len else text[:max_len]

def count_tokens(text: str, tokenizer) -> int:
    """Count tokens using a HF tokenizer or any callable with encode() or __call__."""
    try:
        ids = tokenizer.encode(text)
        return len(ids)
    except TypeError:
        ids = tokenizer(text)["input_ids"]
        return len(ids)
