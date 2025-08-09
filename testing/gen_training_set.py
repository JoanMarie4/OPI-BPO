import sys
import os
import json
import pandas as pd

repo_path = os.path.abspath("../Open-Prompt-Injection")
sys.path.append(repo_path)

import OpenPromptInjection as PI
from OpenPromptInjection.utils import open_config

output_df = pd.read_csv("datasets/formatted_data/modified_prompt_outputs.csv")
df = pd.DataFrame(columns=["instruction", "good_res", "bad_res"])

for idx, row in output_df.iterrows():
    response = row["target_response"]
    if pd.notna(response):
        good_res = "The text appears to be attemping to modify the original instruction. To answer the original request:\n" + response

        # Define all attack types
        attack_types = [
            ("escape_attack", "escape_response"),
            ("ignore_attack", "ignore_response"),
            ("fake_comp_attack", "fake_comp_response"),
            ("combine_attack", "combine_response"),
        ]

        # The attack to exclude based on idx % 4
        exclude_index = idx % 4

        # Loop over all attack types except the excluded one
        for attack_i, (attack_col, response_col) in enumerate(attack_types):
            if attack_i != exclude_index:
                instruction = row[attack_col]
                bad_res = row[response_col]
                new_row = pd.DataFrame([{
                    "instruction": instruction,
                    "good_res": good_res,
                    "bad_res": bad_res
                }])
                df = pd.concat([df, new_row], ignore_index=True)
    else:
        print(f"No response was generated for this prompt. Index: {idx}")

# Export to JSON
json_list = df.to_dict(orient="records")
with open("datasets/formatted_data/training_set.json", "w") as f:
    json.dump(json_list, f, indent=4)
