import argparse
import json
from hashlib import md5

from qdrant_client import QdrantClient
from qdrant_client.models import CollectionStatus, Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

QDRANT_COLLECTION_NAME = "docfinqa_chunks"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
BATCH_SIZE = 64


def batch(iterable, size):
    """Yield successive n-sized chunks from iterable."""
    for i in range(0, len(iterable), size):
        yield iterable[i : i + size]


def connect_qdrant():
    return QdrantClient(host="localhost", port=6334)


def create_collection_if_not_exists(client, vector_dim):
    collections = client.get_collections().collections
    exists = any(col.name == QDRANT_COLLECTION_NAME for col in collections)

    if not exists:
        client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
        )
    else:
        status = client.get_collection(QDRANT_COLLECTION_NAME).status
        if status != CollectionStatus.GREEN:
            print(f"⚠️ Collection exists but is not ready (status: {status}).")


def get_numerical_id_from_chunk_id(chunk_id):
    return int(md5(chunk_id.encode("utf-8")).hexdigest(), 16) % (
        10**12
    )  # 12 dígitos máx.


def main(chunks_path):
    # Load chunked data
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Loaded {len(chunks)} chunks.")

    # Load model
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    if model.device.type != "cuda":
        try:
            model = model.to("cuda")
            print("Moved model to GPU.")
        except Exception as e:
            print(f"Could not move model to GPU: {e}")
    client = connect_qdrant()
    create_collection_if_not_exists(
        client, vector_dim=model.get_sentence_embedding_dimension()
    )

    points = []
    chunk_batches = list(batch(chunks, BATCH_SIZE))
    for chunk_batch in tqdm(chunk_batches, desc="Embedding chunks"):
        texts = [item["chunk_text"] for item in chunk_batch]
        vectors = model.encode(
            texts, convert_to_numpy=True, batch_size=128, show_progress_bar=False
        )

        for vector, item in zip(vectors, chunk_batch):
            points.append(
                PointStruct(
                    id=get_numerical_id_from_chunk_id(item["chunk_id"]),
                    vector=vector.tolist(),
                    payload={
                        "chunk_id": item["chunk_id"],
                        "context_id": item["context_id"],
                        "chunk_text": item["chunk_text"],
                    },
                )
            )
        client.upsert(collection_name=QDRANT_COLLECTION_NAME, points=points)
        points.clear()
    print(f"All chunks indexed in Qdrant collection '{QDRANT_COLLECTION_NAME}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chunks_path", type=str, required=True, help="Path to JSON with chunked data"
    )
    args = parser.parse_args()
    main(args.chunks_path)
