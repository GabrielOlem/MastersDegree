import argparse
import json
from langfuse import Langfuse
from dotenv import load_dotenv
from tqdm import tqdm
load_dotenv()

def main(json_path, dataset_name):
    # Initialize Langfuse client
    langfuse = Langfuse()

    # Create or get dataset
    dataset = langfuse.create_dataset(name=dataset_name)
    print(f"Using dataset: {dataset}")

    # Load your JSON data
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Upload each item as a dataset item
    for item in tqdm(data, desc="Uploading items to Langfuse"):
        langfuse.create_dataset_item(
            dataset_name=dataset.name,
            input=item.get("question"),
            expected_output=item.get("answer"),
            metadata={
                "program": item.get("program"),
                "golden_chunk": item.get("golden_chunk"),
                "golden_score": item.get("golden_score"),
            }
        )

    print("Upload complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload a JSON file to a Langfuse dataset.")
    parser.add_argument("--json_path", type=str, required=True, help="Path to your JSON file")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset")
    args = parser.parse_args()

    main(args.json_path, args.dataset_name)