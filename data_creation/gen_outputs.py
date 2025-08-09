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

    msg = row["target_text"]
    target_df.at[idx, "target_response"] = model.query(msg)

    msg = row["escape_attack"]
    target_df.at[idx, "escape_response"] = model.query(msg)

    msg = row["ignore_attack"]
    target_df.at[idx, "ignore_response"] = model.query(msg)

    msg = row["fake_comp_attack"]
    target_df.at[idx, "fake_comp_response"] = model.query(msg)

    msg = row["combine_attack"]
    target_df.at[idx, "combine_response"] = model.query(msg)


    time.sleep(1)

target_df.to_csv("datasets/gpt_opi_responses.csv", index=False)



# msg = "you find yourself rooting for gai's character to avoid the fate that has befallen every other carmen before her "
# print(model.query(msg))