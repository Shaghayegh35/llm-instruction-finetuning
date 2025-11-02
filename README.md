
# 🧠 LLM Instruction Fine‑Tuning (HF Trainer + PEFT/LoRA)

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-orange?logo=huggingface)
![PEFT](https://img.shields.io/badge/PEFT-LoRA-green)
![License](https://img.shields.io/badge/License-MIT-success)
![Tests](https://img.shields.io/badge/pytest-passing-brightgreen)

Fine‑tune instruction‑following LLMs (e.g., GPT‑2, Mistral‑7B‑Instruct) using **Hugging Face Transformers** and **PEFT/LoRA**.  
Includes configurable pipelines for **training, evaluation, and inference**, plus reproducibility and data formatting helpers.

---

## 🚀 Quickstart

```bash
pip install -r requirements.txt
python train.py --config config.json
python evaluate.py --model_path runs/ft-model --eval_file data/val.jsonl
python infer.py --model_path runs/ft-model --prompt "Explain diffusion models in 2 lines."
```

---

## 🧩 Utility Module — `utils.py`

This module provides essential helpers for **instruction‑style fine‑tuning pipelines**:  
- **Typed validation:** `Record` dataclass ensures dataset consistency.  
- **Prompt generation:** `format_example()` for instruction/input/response formatting.  
- **Reproducibility:** `set_global_seed()` controls randomness across libraries.  
- **Token management:** `count_tokens()` for model‑agnostic token counting.  
- **String safety:** `safe_truncate()` prevents overflow in long text.

**Prompt Examples:**

_With Input:_
```text
### Instruction:
Summarize the paragraph.

### Input:
Neural networks are powerful...

### Response:
They are deep learning models that learn complex patterns.
```

_Without Input:_
```text
### Instruction:
Translate to French.

### Response:
Bonjour, comment ça va ?
```

---

## 🧪 Testing

```bash
pytest tests/test_utils.py -q
```

---

## 🧱 Example Integration

```python
from utils import format_example, set_global_seed

set_global_seed(42)

sample = {
    "instruction": "Summarize this text",
    "input": "Large language models can follow instructions.",
    "output": "They can follow human commands using fine-tuning."
}

print(format_example(sample))
```

---

## 📄 License

This project is released under the [MIT License](./LICENSE).  
Created and maintained by **Shaghayegh Khalighiyan**.

---
