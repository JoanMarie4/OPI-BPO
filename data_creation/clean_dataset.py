import json

def has_bad_request_error(entry):
    origin = entry.get("origin", {})
    for key in ["instruction", "good_res", "bad_res"]:
        val = origin.get(key, "")
        if isinstance(val, str) and "BadRequestError" in val:
            return True
    return False

input_path = 'datasets/formatted_data/better_instructions/dataset_optimized.jsonl'
output_path = 'datasets/formatted_data/better_instructions/dataset_filtered.jsonl'

with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
    kept, removed = 0, 0
    for line in infile:
        try:
            data = json.loads(line)
            if not has_bad_request_error(data):
                json.dump(data, outfile)
                outfile.write('\n')
                kept += 1
            else:
                removed += 1
        except json.JSONDecodeError:
            print("Skipping malformed line.")
            removed += 1

print(f"Done. Kept: {kept}, Removed: {removed}")