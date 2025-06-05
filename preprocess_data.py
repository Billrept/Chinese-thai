import json
import argparse
from pathlib import Path

def preprocess_jsonl(input_path: str, output_path: str, use_context: bool = False):
    in_path = Path(input_path)
    out_path = Path(output_path)
    assert in_path.exists(), f"Input file {in_path} not found."

    with in_path.open("r", encoding="utf-8") as fin, \
         out_path.open("w", encoding="utf-8") as fout:

        for line in fin:
            obj = json.loads(line)
            src = obj.get("source", "").strip()
            tgt = obj.get("translation", "").strip()
            if not src or not tgt:
                continue

            if use_context:
                # Flatten newlines in context to a single space
                ctx = obj.get("context", "").replace("\n", " ").strip()
                prompt = f"Conversation context: {ctx} Translate the next Chinese sentence to Thai: {src}"
            else:
                prompt = f"Translate Chinese to Thai: {src}"

            out_obj = {"prompt": prompt, "target": tgt}
            fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

    print(f"Preprocessed {input_path} → {output_path}. Entries: {sum(1 for _ in out_path.open())}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess JSONL for translation fine-tuning.")
    parser.add_argument("--input",    type=str, required=True, help="Path to original JSONL file.")
    parser.add_argument("--output",   type=str, required=True, help="Path to write processed JSONL.")
    parser.add_argument("--use_context", action="store_true", help="Whether to prepend context field in prompt.")
    args = parser.parse_args()

    preprocess_jsonl(args.input, args.output, use_context=args.use_context)
