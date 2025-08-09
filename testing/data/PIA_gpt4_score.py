import requests
import multiprocessing
from multiprocessing import Manager
import json
from tqdm import tqdm
import os
import time
import pandas as pd
import random
import argparse

API_KEY = 'Your-API-Key'

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

API_URL = "https://api.openai.com/v1/chat/completions"

def chat_gpt(messages, counter, error_count):
    responses = []
    for i, m in enumerate(messages):
        try:
            message = m['message']
            data = json.dumps({"model": "gpt-4", "messages": message, 'temperature': 0.0})
            response = requests.post(API_URL, headers=HEADERS, data=data)
            response_json = response.json()
            print(response_json)
            res = response_json['choices'][0]['message']['content']
            m['response'] = res
            # save to file
            with open(output_file, 'a', encoding='utf-8') as f:
                print(json.dumps(m, ensure_ascii=False), file=f)

            responses.append(response_json)

            counter += 1
        except Exception as e:
            error_count += 1
            print(e)
        print('running time:{} finished number:{} skipped number:{}'.format(time.time()-s_time, counter, error_count), end='\r')

    return responses


def get_messages_list():

    if task_name.count("test_set") or task_name.count("dolly"):
        idx = "idx"
    elif task_name.count("self_instruct"):
        idx = "id"
    elif task_name.count("vicuna"):
        idx = "question_id"
    else:
        print("idx Not implemented")
        assert False

    evaluated = []
    if os.path.exists(output_file):
        with open(output_file, encoding='utf-8') as f:
            lines = f.readlines()
        for i in lines:
            evaluated.append(json.loads(i)['origin'])

    with open(input_file_a) as f:
        d_a = json.load(f)

    with open(input_file_b) as f:
        d_b = json.load(f)

    # Load the preferred (good) responses file
    with open(preferred_file) as f:
        d_pref = json.load(f)

    messages_list = []

    for i, j, pref in zip(d_a, d_b, d_pref):
        assert (i[idx] == j[idx] == pref[idx])
        if i[idx] in evaluated:
            continue

        # Randomize order of A/B to avoid position bias
        if random.randint(0, 1) == 0:
            option_a = i
            res_a = i['res']
            res_b = j['res']
        else:
            option_a = j
            res_a = j['res']
            res_b = i['res']

        if task_name.count("self_instruct") or task_name.count("dolly"):
            question = (i['instruction'] + '\n' + i['context']).strip()
        elif task_name.count("test_set"):
            question = i['context'].strip()
        elif task_name.count("vicuna"):
            question = i['text'].strip()
        else:
            print("Not implemented")
            assert False

        preferred_res = pref['res']

        messages_list.append({
            'message': [
                {"role": 'system', "content": prompt['system_prompt']},
                {"role": "user", "content": prompt['prompt_template']
                    .replace('{question}', question)
                    .replace('{preferred_response}', preferred_res)
                    .replace('{answer_a}', res_a)
                    .replace('{answer_b}', res_b)
                }
            ],
            'origin': i[idx],
            'option_a': option_a,
        })

    return messages_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_file_a', type=str)
    parser.add_argument('--input_file_b', type=str)
    parser.add_argument('--preferred_file', type=str, help="Path to file with good/reference responses")
    parser.add_argument('--task_name', type=str)
    parser.add_argument('--output_file', type=str)
    args = parser.parse_args()  
    
    input_file_a = args.input_file_a
    input_file_b = args.input_file_b
    preferred_file = args.preferred_file
    task_name = args.task_name
    output_file = args.output_file

    prompt = {
        "name": "pair-v2-ref-align",
        "type": "pairwise",
        "system_prompt": (
            "Please act as an impartial judge and evaluate which of two AI assistant responses "
            "best matches the style, content, and instruction fulfillment of a given preferred (reference) response. "
            "You will be given: the original user question, the preferred response, and two candidate responses. "
            "Your decision should be based on:\n"
            "1. How well the candidate fulfills the original instruction in the same way as the preferred response.\n"
            "2. Whether it includes all important information present in the preferred response.\n"
            "3. Whether it avoids adding unrelated answers or extra information not found in the preferred response.\n"
            "4. Overall helpfulness, accuracy, and clarity only as they relate to the preferred response's scope.\n\n"
            "Do not reward verbosity or penalize brevity if the meaning is preserved.\n"
            "Ignore minor wording or formatting differences.\n"
            "Avoid position bias — the order of responses should not affect your choice.\n"
            "After your short explanation, output your final verdict strictly as:\n"
            "\"[[A]]\" if Assistant A is better aligned,\n"
            "\"[[B]]\" if Assistant B is better aligned,\n"
            "\"[[C]]\" if both are equally aligned."
        ),
        "prompt_template": (
            "[User Question]\n{question}\n\n"
            "[Preferred Response]\n{preferred_response}\n\n"
            "[The Start of Assistant A's Answer]\n{answer_a}\n[The End of Assistant A's Answer]\n\n"
            "[The Start of Assistant B's Answer]\n{answer_b}\n[The End of Assistant B's Answer]"
        ),
        "description": "Prompt for comparing two responses to see which better matches a preferred reference answer",
        "category": "general",
        "output_format": "[[A]]"
    }

    if not os.path.exists(output_file):
        open(output_file, 'w').close()

    messages_list = get_messages_list()
    print("total num: ", len(messages_list))
    s_time = time.time()
    responses = chat_gpt(messages_list, 0, 0)
