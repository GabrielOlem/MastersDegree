import argparse
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, SearchRequest
from hashlib import md5

EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
COLLECTION_NAME = "docfinqa_chunks"

def main(questions_path, output_path):
    # Carrega as perguntas (dev.json com question, answer, etc)
    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    model = SentenceTransformer(EMBEDDING_MODEL)
    client = QdrantClient(host="localhost", port=6333)
    print("Modelo e Qdrant conectados.")

    golden_dataset = []
    for q in tqdm(questions, desc="Processando questões"):
        question = q["Question"]
        answer = q["Answer"]
        program = q.get("Program", "")
        context = q["Context"]  # usado como filtro

        context_hash = md5(context.encode('utf-8')).hexdigest()  # mesmo valor usado no embed_chunks_to_qdrant.py

        q_embedding = model.encode(question).tolist()
        search_result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=q_embedding,
            limit=1,
            with_payload=True,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="context_id",
                        match=MatchValue(value=context_hash)
                    )
                ]
            )
        )
        if not search_result:
            print(f"Nenhum chunk encontrado para question={q['Question']}")
            continue

        best_chunk = search_result.points[0].payload["chunk_text"]
        
        golden_dataset.append({
            #"question_id": q.get("question_id", None),
            "question": question,
            "answer": answer,
            "program": program,
            "golden_chunk": best_chunk,
            "golden_score": search_result.points[0].score
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(golden_dataset, f, ensure_ascii=False, indent=2)
        print(f"Salvo {len(golden_dataset)} golden chunks em {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gera golden chunks via Qdrant.")
    parser.add_argument("--questions_path", type=str, required=True, help="Caminho para o arquivo dev.json")
    parser.add_argument("--output_path", type=str, required=True, help="Arquivo JSON de saída com golden chunks")
    args = parser.parse_args()
    main(args.questions_path, args.output_path)
