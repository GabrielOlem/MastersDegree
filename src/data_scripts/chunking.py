import argparse
import json
from tqdm import tqdm
from transformers import AutoTokenizer
from hashlib import md5
from qdrant_client import QdrantClient
from qdrant_client.http import models

CHUNK_SIZE = 512
CHUNK_OVERLAP = 128
TOKENIZER_NAME = "bert-base-uncased"
QDRANT_COLLECTION_NAME = "docfinqa_chunks"

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

def context_exists_in_qdrant(qdrant_client, collection_name, context_id):
    """Check if a context_id is already indexed in Qdrant."""
    count = qdrant_client.count(
        collection_name=collection_name,
        count_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="context_id",
                    match=models.MatchValue(value=context_id)
                )
            ]
        )
    ).count
    return count > 0

def main(input_path, output_path):
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    print('Tokenizer ready')
    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    print("Tokenizer and data ready")

    qdrant = QdrantClient(host="localhost", port=6334)
    all_chunks = []
    seen_contexts = set()

    for item in tqdm(raw_data, desc="Processing unique contexts"):
        context = item["Context"]
        context_hash = md5(context.encode('utf-8')).hexdigest()
        if context_hash in seen_contexts:
            print(f"Skipping duplicate context: {context_hash}")
            continue
        seen_contexts.add(context_hash)

        if context_exists_in_qdrant(qdrant, QDRANT_COLLECTION_NAME, context_hash):
            print(f"Context {context_hash} already exists in Qdrant, skipping.")
            continue

        chunks = chunk_text(context, tokenizer, CHUNK_SIZE, CHUNK_OVERLAP)

        for idx, chunk in enumerate(chunks):
            all_chunks.append({
                "context_id": context_hash,
                "chunk_id": f"{context_hash}_chunk_{idx}",
                "chunk_text": chunk
            })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(all_chunks)} chunks to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate chunks from input JSON.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to input JSON file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to output JSON file")
    args = parser.parse_args()
    print('a')
    main(args.input_path, args.output_path)
