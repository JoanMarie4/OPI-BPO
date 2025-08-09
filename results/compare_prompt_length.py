import json
import tiktoken

file_path = "datasets/formatted_data/better_instructions/dataset_optimized.json"
with open(file_path, "r") as f:
    data = json.load(f)

# === Setup tokenizer ===
enc = tiktoken.encoding_for_model("gpt-4")  # or use "gpt-3.5-turbo" if needed

total_prompt_len = 0
total_opt_len = 0
total_prompt_tokens = 0
total_opt_tokens = 0
count = 0


for d in data:
    prompt = d.get("prompt", "")
    optimized = d.get("optimized_prompt", "")

    total_prompt_len += len(prompt)
    total_opt_len += len(optimized)
    total_prompt_tokens += len(enc.encode(prompt))
    total_opt_tokens += len(enc.encode(optimized))
    count += 1


avg_prompt_len = total_prompt_len / count
avg_opt_len = total_opt_len / count
avg_prompt_tokens = total_prompt_tokens / count
avg_opt_tokens = total_opt_tokens / count

print(f"Average Prompt Length (chars): {avg_prompt_len:.2f}")
print(f"Average Optimized Prompt Length (chars): {avg_opt_len:.2f}")
print(f"Average Prompt Token Count (exact): {avg_prompt_tokens:.2f}")
print(f"Average Optimized Prompt Token Count (exact): {avg_opt_tokens:.2f}")
