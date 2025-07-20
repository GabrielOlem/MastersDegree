import argparse
import json
import os
import time

import openai
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv(override=True)

MODEL = "gpt-4o"
API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = API_KEY


def build_prompt(question, context):
    return f"""
    You are a financial reasoning assistant.

    Given the following financial document and a question, 
    Write a Python-style program that calculates the answer using variables and arithmetic operations.

    Only use the information provided in the document.

    End your program with a line that assigns the final result to the variable `answer`.
    
    ### Document:
    Context: {context}

    ### Question:
    Question: {question}

    ### Python-style Program:
    """


def get_response_from_openai(prompt, model=MODEL, temperature=0.3):
    try:
        response = openai.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error: {e}")
        return ""


def main(input_path, output_path):
    with open(input_path, "r") as f:
        data = json.load(f)

    output_data = []

    for item in tqdm(data, desc="Processing items"):
        question = item["question"]
        context = item["golden_chunk"]
        answer = item["answer"]
        golden_program = item.get("golden_program", "")

        prompt = build_prompt(question, context)
        golden_program_gen = get_response_from_openai(prompt)

        output_data.append(
            {
                "question_id": item["question_id"],
                "question": question,
                "golden_chunk": context,
                "answer": answer,
                "golden_program_generated": golden_program_gen,
                "golden_program": golden_program,
            }
        )

        time.sleep(1.5)  # Rate limiting

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=4)
    print(f"Rationale generation completed. Output saved to {output_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate rationales using OpenAI API."
    )
    parser.add_argument(
        "--input_path", type=str, required=True, help="Path to input JSON file"
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="Path to output JSON file"
    )
    args = parser.parse_args()
    main(args.input_path, args.output_path)
