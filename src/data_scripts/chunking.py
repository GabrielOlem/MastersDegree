import argparse
import json
from tqdm import tqdm
from transformers import AutoTokenizer
from hashlib import md5

CHUNK_SIZE = 512
CHUNK_OVERLAP = 128
TOKENIZER_NAME = "bert-base-uncased"

def chunk_text(text, tokenizer, chunk_size, overlap):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)
        start += chunk_size - overlap
    return chunks

def main(input_path, output_path):
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    print("Tokenizer and data ready")
    all_chunks = []
    seen_contexts = set()

    for item in tqdm(raw_data, desc="Processing unique contexts"):
        context = item["Context"]
        context_hash = md5(context.encode('utf-8')).hexdigest()
        if context_hash in seen_contexts:
            print(f"Skipping duplicate context: {context_hash}")
            continue

        chunks = chunk_text(context, tokenizer, CHUNK_SIZE, CHUNK_OVERLAP)

        for idx, chunk in enumerate(chunks):
            all_chunks.append({
                "context_id": context_hash,
                "chunk_id": f"{context_hash}_chunk_{idx}",
                "chunk_text": chunk
            })
        seen_contexts.add(context_hash)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(all_chunks)} chunks to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate chunks from input JSON.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to input JSON file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to output JSON file")
    args = parser.parse_args()
    main(args.input_path, args.output_path)
