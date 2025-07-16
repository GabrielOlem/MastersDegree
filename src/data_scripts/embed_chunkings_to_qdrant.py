import argparse
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance, CollectionStatus

QDRANT_COLLECTION_NAME = "docfinqa_chunks"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
BATCH_SIZE = 100


def batch(iterable, size):
    """Yield successive n-sized chunks from iterable."""
    for i in range(0, len(iterable), size):
        yield iterable[i:i + size]

def connect_qdrant():
    return QdrantClient(host="localhost", port=6333)

def create_collection_if_not_exists(client, vector_dim):
    collections = client.get_collections().collections
    exists = any(col.name == QDRANT_COLLECTION_NAME for col in collections)

    if not exists:
        client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE)
        )
    else:
        status = client.get_collection(QDRANT_COLLECTION_NAME).status
        if status != CollectionStatus.GREEN:
            print(f"⚠️ Collection exists but is not ready (status: {status}).")

def main(chunks_path):
    # Load chunked data
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Loaded {len(chunks)} chunks.")

    # Load model
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    client = connect_qdrant()
    create_collection_if_not_exists(client, vector_dim=model.get_sentence_embedding_dimension())

    points = []
    for idx, item in enumerate(tqdm(chunks, desc="Embedding & indexing")):
        vector = model.encode(item['chunk_text']).tolist()
        points.append(PointStruct(
            id=idx,
            vector=vector,
            payload={
                "chunk_id": item["chunk_id"],
                "context_id": item["context_id"],
                "chunk_text": item['chunk_text']
            }
        ))


    for batch_points in tqdm(batch(points, BATCH_SIZE), desc="Sending to Qdrant"):
        client.upsert(
            collection_name=QDRANT_COLLECTION_NAME,
            points=batch_points
        )
    print(f"All chunks indexed in Qdrant collection '{QDRANT_COLLECTION_NAME}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks_path", type=str, required=True, help="Path to JSON with chunked data")
    args = parser.parse_args()
    main(args.chunks_path)