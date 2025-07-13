import argparse
import nltk
nltk.download("punkt")
nltk.download("punkt_tab")

import json
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

def main(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        chunk_data = json.load(f)

    chunks_by_question = defaultdict(list)
    for chunk in chunk_data:
        chunks_by_question[chunk["question_id"]].append(chunk)

    model = SentenceTransformer(EMBEDDING_MODEL)
    print("Model and data ready")
    golden_dataset = []
    for qid, chunks in tqdm(chunks_by_question.items(), desc="Processing questions"):
        question = chunks[0]["question"]
        answer = chunks[0]["answer"]
        golden_program = chunks[0].get("golden_program")

        q_embedding = model.encode(question, convert_to_tensor=True)

        chunk_tests = [chunk["chunk_text"] for chunk in chunks]
        chunk_embeddings = model.encode(chunk_tests, convert_to_tensor=True)
        
        similarities = util.cos_sim(q_embedding, chunk_embeddings)[0]
        best_idx = similarities.argmax().item()
        best_chunk = chunks[best_idx]

        golden_dataset.append({
            "question_id": qid,
            "question": question,
            "answer": answer,
            "golden_program": golden_program,
            "golden_chunk": best_chunk["chunk_text"],
            "score": similarities[best_idx].item()
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(golden_dataset, f, ensure_ascii=False, indent=4)
        print(f"Saved {len(golden_dataset)} golden chunks to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate golden chunks from chunked data.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to input chunked JSON file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to output golden JSON file")
    args = parser.parse_args()
    main(args.input_path, args.output_path)