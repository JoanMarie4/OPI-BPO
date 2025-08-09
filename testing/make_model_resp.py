import json
import pandas as pd

# Load the original dataset
with open("data/bpo_model_output.json", "r") as f:
    original_data = json.load(f)

# Create a new DataFrame with renamed keys
df = pd.DataFrame(original_data)
df = df.rename(columns={"instruction": "question", "bad_res": "answer"})

# Keep only the 'prompt' and 'response' columns
df = df[["question", "answer"]]

# Convert to list of dicts
converted_data = df.to_dict(orient="records")

# Save to JSONL file
with open("data/og_model_resps.jsonl", "w", encoding="utf-8") as f:
    for entry in converted_data:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
