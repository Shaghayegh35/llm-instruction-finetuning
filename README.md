# LLM Instruction Fine‑Tuning (HF Trainer + PEFT/LoRA)

![license](https://img.shields.io/badge/license-MIT-green)
![hf](https://img.shields.io/badge/Transformers-4.44+-blue)
![peft](https://img.shields.io/badge/PEFT-LoRA-orange)

Production‑ready template to fine‑tune an instruction‑following model (e.g., `gpt2`, `mistralai/Mistral-7B-Instruct`) using **Hugging Face Transformers** and **PEFT/LoRA**.  
Includes evaluation, inference, and MLflow‑ready hooks.

> Created 2025-11-02

## Folder
```
llm-instruction-finetuning/
├─ data/                 # small toy jsonl train/val
├─ train.py              # Trainer + LoRA
├─ evaluate.py           # prints generations + ROUGE‑L (can extend)
├─ infer.py              # single‑prompt inference
├─ utils.py              # prompt format
├─ config.json           # training config
└─ requirements.txt
```

## Quickstart
```bash
pip install -r requirements.txt
python train.py --config config.json
python evaluate.py --model_path runs/ft-model --eval_file data/val.jsonl
python infer.py --model_path runs/ft-model --prompt "Summarize YOLOv9 in 2 bullets."
```

## Examples

**Train**
```bash
python train.py --config config.json
```

**Evaluate**
```bash
python evaluate.py --model_path runs/ft-model --eval_file data/val.jsonl
```

**Infer**
```bash
python infer.py --model_path runs/ft-model --prompt "Explain diffusion models in 2 lines."
```


## Examples

**Train**
```bash
python train.py --config config.json
```

**Evaluate**
```bash
python evaluate.py --model_path runs/ft-model --eval_file data/val.jsonl
```

**Infer**
```bash
python infer.py --model_path runs/ft-model --prompt "Explain diffusion models in 2 lines."
```

