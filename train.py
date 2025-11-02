import json, argparse
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from utils import format_example

def load_jsonl(p):
    rows = [json.loads(l) for l in open(p, "r", encoding="utf-8") if l.strip()]
    return Dataset.from_list(rows)

def tokenize(batch, tok, max_len):
    texts = [format_example(x) for x in batch["__raw__"]]
    return tok(texts, truncation=True, max_length=max_len)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    args = ap.parse_args()
    cfg = json.load(open(args.config))

    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg["base_model"])

    lora = LoraConfig(r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"], task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)

    train = load_jsonl("data/train.jsonl")
    val = load_jsonl("data/val.jsonl")
    train = train.add_column("__raw__", [dict(x) for x in train])
    val = val.add_column("__raw__", [dict(x) for x in val])
    train = train.map(lambda x: tokenize(x, tok, cfg["max_length"]), batched=True, remove_columns=train.column_names)
    val = val.map(lambda x: tokenize(x, tok, cfg["max_length"]), batched=True, remove_columns=val.column_names)

    args_tr = TrainingArguments(
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=cfg["train_batch_size"],
        per_device_eval_batch_size=cfg["eval_batch_size"],
        learning_rate=cfg["learning_rate"],
        evaluation_strategy="steps", eval_steps=50, save_steps=50, logging_steps=10,
        fp16=False, report_to=[]
    )
    collator = DataCollatorForLanguageModeling(tok, mlm=False)
    tr = Trainer(model=model, args=args_tr, train_dataset=train, eval_dataset=val, data_collator=collator, tokenizer=tok)
    tr.train()
    tr.save_model(cfg["output_dir"])
    tok.save_pretrained(cfg["output_dir"])

if __name__ == "__main__":
    main()
