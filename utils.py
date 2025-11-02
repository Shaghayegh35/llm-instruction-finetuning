import json
def format_example(rec):
    instr = rec.get("instruction","").strip()
    inp = rec.get("input","").strip()
    out = rec.get("output","").strip()
    if inp:
        return f"### Instruction:\n{instr}\n\n### Input:\n{inp}\n\n### Response:\n{out}"
    else:
        return f"### Instruction:\n{instr}\n\n### Response:\n{out}"
