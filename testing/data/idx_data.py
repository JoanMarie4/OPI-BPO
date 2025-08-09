import json

def add_idx_to_file(input_path, output_path):
    # Detect if file is JSONL or JSON array by trying to load the entire file as JSON
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)  # Try JSON array
        is_json_array = True
    except json.JSONDecodeError:
        is_json_array = False

    if is_json_array:
        # data is a list of dicts
        for i, entry in enumerate(data):
            entry['idx'] = i
        # Write as JSONL
        with open(output_path, "w", encoding="utf-8") as f_out:
            for entry in data:
                f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")
    else:
        # Assume JSONL: one JSON object per line
        updated_entries = []
        with open(input_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                entry = json.loads(line)
                entry['idx'] = i
                updated_entries.append(entry)
        with open(output_path, "w", encoding="utf-8") as f_out:
            for entry in updated_entries:
                f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    input_file = "good_resps.jsonl"      # Change this to your input filename
    output_file = "responses.jsonl"  # Change this to desired output filename

    add_idx_to_file(input_file, output_file)
    print(f"Added 'idx' to each entry and saved to {output_file}")
