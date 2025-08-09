import sys
import os
import time

repo_path = os.path.abspath("../Open-Prompt-Injection")
sys.path.append(repo_path)

import OpenPromptInjection as PI
from OpenPromptInjection.utils import open_config
import pandas as pd

# Login using e.g. `huggingface-cli login` to access this dataset
target_df = pd.read_csv("hf://datasets/xxz224/prompt-injection-attack-dataset/complete_dataset.csv")
target_df["target_response"] = None
target_df["escape_response"] = None
target_df["ignore_response"] = None
target_df["fake_comp_response"] = None
target_df["combine_response"] = None


model_config_path = "../Open-Prompt-Injection/configs/model_configs/gpt_config.json"
model_config = open_config(config_path=model_config_path)
model = PI.create_model(config=model_config)
model.print_model_info()

for idx, row in target_df.iterrows():
    print(f"At Index: {idx}") 
    task_type = row["target_task_type"]
    prompt = f"Can you do {task_type} on this text: "

    # target_text
    msg = prompt + row["target_text"]
    target_df.at[idx, "target_text"] = msg
    target_df.at[idx, "target_response"] = model.query(msg)

    # escape_attack
    msg = prompt + row["escape_attack"]
    target_df.at[idx, "escape_attack"] = msg
    target_df.at[idx, "escape_response"] = model.query(msg)

    # ignore_attack
    msg = prompt + row["ignore_attack"]
    target_df.at[idx, "ignore_attack"] = msg
    target_df.at[idx, "ignore_response"] = model.query(msg)

    # fake_comp_attack
    msg = prompt + row["fake_comp_attack"]
    target_df.at[idx, "fake_comp_attack"] = msg
    target_df.at[idx, "fake_comp_response"] = model.query(msg)

    # combine_attack
    msg = prompt + row["combine_attack"]
    target_df.at[idx, "combine_attack"] = msg
    target_df.at[idx, "combine_response"] = model.query(msg)

target_df.to_csv("data/gpt_opi_responses.csv", index=False)

