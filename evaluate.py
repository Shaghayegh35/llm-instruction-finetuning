import json, argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--eval_file", required=True)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    for line in open(args.eval_file, "r", encoding="utf-8"):
        ex = json.loads(line)
        prompt = f"### Instruction:\n{ex['instruction']}\n\n### Input:\n{ex.get('input','')}\n\n### Response:\n"
        out = model.generate(**tok(prompt, return_tensors='pt'), max_new_tokens=64)
        print(tok.decode(out[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
