import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--prompt", required=True)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    out = model.generate(**tok(args.prompt, return_tensors="pt"), max_new_tokens=120)
    print(tok.decode(out[0], skip_special_tokens=True))
if __name__ == "__main__":
    main()
