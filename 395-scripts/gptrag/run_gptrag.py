import json
import time
import os
from openai import OpenAI

API_KEY = os.environ.get("OPENAI_API_KEY")
if API_KEY is None:
    raise ValueError("Environment variable OPENAI_API_KEY is not set.")

client = OpenAI(api_key=API_KEY)

INPUT_FILE = "./Datasets/Questions/novel_questions.json"
OUTPUT_FILE = "./395-scripts/gptrag/results/novel_results_5nanogpt.json"
MODEL_NAME = "gpt-5-nano"

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

results = []
total = len(data)

for idx, item in enumerate(data, start=1):
    percent = (idx / total) * 100
    print(f"Processing {idx}/{total} ({percent:.1f}%) -- {item['id']}")

    question = item["question"]

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Answer factually, clearly, and concisely."},
                {"role": "user", "content": question}
            ],
            temperature=1.0
        )
        model_answer = response.choices[0].message.content
    except Exception as e:
        print(f"Error for {item['id']}: {e}")
        model_answer = ""

    entry = {
        "id": item.get("id"),
        "question": item.get("question"),
        "source": item.get("source"),
        "question_type": item.get("question_type"),
        "evidence": item.get("evidence"),
        "ground_truth": item.get("ground_truth"),
        "answer": model_answer,
        "data": {"chunks": []}
    }

    results.append(entry)
    time.sleep(0.2)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print(f"\nDone! Wrote {len(results)} items to {OUTPUT_FILE}")
