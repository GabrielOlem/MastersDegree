import argparse
from src.data_scripts.embed_chunkings_to_qdrant import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run embed chunkings pipeline.")
    parser.add_argument("--chunks_path", type=str, required=True, help="Path to input JSON file")
    args = parser.parse_args()
    main(args.chunks_path)