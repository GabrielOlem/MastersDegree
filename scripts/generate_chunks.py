import argparse
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
import json
from tqdm import tqdm

CHUNK_SIZE = 1024
CHUNK_OVERLAP = 200

def main(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    parser = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    all_chunks = []
    for doc_id, item in enumerate(tqdm(raw_data, desc="Processing items")):
        paragraph_text = item['Context']
        doc = Document(
            text=paragraph_text,
            metadata={'question_id': doc_id}
        )
        nodes = parser.get_nodes_from_documents([doc])

        print(len(nodes))
        for idx, node in enumerate(nodes):
            chunk = {
                'question_id': doc_id,
                'chunk_id': f"{doc_id}_chunk_{idx}",
                'chunk_text': node.get_content(),
                'question': item['Question'],
                'answer': item['Answer'],
                'golden_program': item['Program']
            }
            all_chunks.append(chunk)
        print(all_chunks[-1])

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=4)
        print(f"Saved {len(all_chunks)} chunks to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate chunks from input JSON.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to input JSON file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to output JSON file")
    args = parser.parse_args()
    main(args.input_path, args.output_path)