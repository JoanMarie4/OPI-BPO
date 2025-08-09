# @Time    : 2023/4/2 22:49
# @Author  : tk
# @FileName: infer

import torch
import os
import sys

repo_path = os.path.abspath("../../Open-Prompt-Injection")
sys.path.append(repo_path)

import OpenPromptInjection as PI
from OpenPromptInjection.utils import open_config
import pandas as pd


repo_path = os.path.abspath("../../BPO/src/training")
sys.path.append(repo_path)

from deep_training.data_helper import ModelArguments
from transformers import HfArgumentParser, AutoConfig
from data_utils import train_info_args, NN_DataHelper, get_deepspeed_config,build_template
from aigc_zoo.model_zoo.llm.llm_model import MyTransformer
from aigc_zoo.utils.llm_generate import Generate
import json
from tqdm import tqdm

deep_config = get_deepspeed_config()


gpt_model_config_path = "../../Open-Prompt-Injection/configs/model_configs/gpt_config.json"
gpt_model_config = open_config(config_path=gpt_model_config_path)
gpt_model = PI.create_model(config=gpt_model_config)
gpt_model.print_model_info()


if __name__ == '__main__':

    input_file = 'small_test.json'
    output_file = 'bpo_model_output.json'

    parser = HfArgumentParser((ModelArguments,))
    (model_args,) = parser.parse_dict(train_info_args, allow_extra_keys=True)

    dataHelper = NN_DataHelper(model_args)
    tokenizer, _, _,_= dataHelper.load_tokenizer_and_config()
    

    config = AutoConfig.from_pretrained('../../BPO/src/training/output/best_ckpt')
    pl_model = MyTransformer(config=config, model_args=model_args,torch_dtype=config.torch_dtype,)

    train_weight = '../../BPO/src/training/output/final_model/pytorch_model.bin'
    
    
    pl_model.load_sft_weight(train_weight,strict=True)

    model = pl_model.get_llm_model()

    model.eval().half().cuda()


    with open(input_file, encoding='utf-8') as f:
        text_list = json.load(f)[:]
    
    gen_res = []

    for idx, input in enumerate(tqdm(text_list[:])):
        try:
            query = build_template((input['instruction']).strip())
            max_new_tokens = 512

            tokens = tokenizer(query, return_tensors='pt')
            seq_len = tokens['input_ids'].size(1)
            max_id  = tokens['input_ids'].max().item()

            print(f"[DEBUG] idx={idx}  seq_len={seq_len}  max_token_id={max_id}")

            if seq_len + max_new_tokens > config.n_positions:
                print(f"[SKIP] too long: {seq_len}+{max_new_tokens} > {config.n_positions}")
                input['gen_prompt'] = "[INPUT TOO LONG]"
                gen_res.append(input)
                continue
            if max_id >= config.vocab_size:
                print(f"[SKIP] bad token id {max_id} ≥ {config.vocab_size}")
                input['gen_prompt'] = "[BAD TOKEN]"
                gen_res.append(input)
                continue

            response = Generate.generate(
                model,
                query=query,
                tokenizer=tokenizer,
                max_new_tokens=512,
                eos_token_id=config.eos_token_id,
                do_sample=True,
                top_p=0.9,
                temperature=0.6,
                num_beams=1

            )
            input['gen_prompt'] = response.strip()
            input['gen_res'] = gpt_model.query(input['gen_prompt'])


        except Exception as e:
            print(f"\n[Error] Failed on index {idx}")
            print("Instruction:", input.get('instruction'))
            print("Exception:", e)
            input['gen_prompt'] = "[GENERATION ERROR]"

        gen_res.append(input)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(gen_res, f, indent=4, ensure_ascii=False)
        json.dump(gen_res, f, indent=4, ensure_ascii=False)

