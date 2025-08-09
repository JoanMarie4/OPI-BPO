import sys
import os
import json

repo_path = os.path.abspath("../../Open-Prompt-Injection")
sys.path.append(repo_path)

import OpenPromptInjection as PI
from OpenPromptInjection.utils import open_config
import pandas as pd

output_df = pd.read_csv("../datasets/formatted_data/modified_prompt_outputs.csv")
df = pd.DataFrame(columns=["instruction", "good_res", "bad_res"])
instruction =  ""
good_res = ""
bad_res = ""

for idx, row in output_df.iterrows():

    response = row["target_response"]
    if pd.notna(response):
        good_res = "The text appears to be attemping to modify the origianl instruction. To answer the original request:\n" + response
        
        if idx % 4 == 1:
            # escape attack
            instruction = row["escape_attack"]
            bad_res = row["escape_response"]
            new_row = pd.DataFrame([{
            "instruction": instruction,
            "good_res": good_res,
            "bad_res": bad_res
            }])
            df = pd.concat([df, new_row], ignore_index=True)
            
        elif idx % 4 == 2:
            # ignore attack
            instruction = row["ignore_attack"]
            bad_res = row["ignore_response"]
            new_row = pd.DataFrame([{
            "instruction": instruction,
            "good_res": good_res,
            "bad_res": bad_res
            }])
            df = pd.concat([df, new_row], ignore_index=True)

        elif idx % 4 == 3:
            
            #f ake comp attack
            instruction = row["fake_comp_attack"]
            bad_res = row["fake_comp_response"]
            new_row = pd.DataFrame([{
            "instruction": instruction,
            "good_res": good_res,
            "bad_res": bad_res
            }])
            df = pd.concat([df, new_row], ignore_index=True)

        else:
            # combine attack
            instruction = row["combine_attack"]
            bad_res = row["combine_response"]
            new_row = pd.DataFrame([{
            "instruction": instruction,
            "good_res": good_res,
            "bad_res": bad_res
            }])
            df = pd.concat([df, new_row], ignore_index=True)
    else:
        print("No response was generated for this prompt.")
        print(f"Index: {idx}, Response: {response}")



json_list = df.to_dict(orient="records")

with open("small_test.json", "w") as f:
    json.dump(json_list, f, indent=4)
