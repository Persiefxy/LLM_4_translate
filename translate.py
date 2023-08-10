import pandas as pd
import json
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import torch
import argparse
import os
import fire
from datetime import datetime

# python translate.py -Model LlaMa --Dataset ./dataset/wmt22en_zh/en_zh.json --Output result --Ref ref --Output_dir ./translate_result/
def read(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return lines
def main(Model:str, Dataset:str, Output:str, Ref:str, Output_dir:str):
    os.makedirs(Output_dir, exist_ok=True)
    current_time = datetime.now()
    datetime_string = current_time.strftime('%Y-%m-%d_%H:%M:%S')
    os.makedirs(Output_dir+Model+'/', exist_ok=True)
    output = Output_dir+Model+'/'+datetime_string
    text = Dataset.split('.')[-1]
    with open(Dataset, 'r') as f:
        assert text in ['json'], "please input json file"
        file = json.load(f)
    #读取模型
    if(Model == 'GLM'):
        model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().cuda()
        tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
        response, history = model.chat(tokenizer ,"You are a professional translator, especially good at translating from English to Chinese, Translations should be streamlined and realized in one sentence as much as possible.", history=[])
    elif(Model == 'LlaMa'):
        model = AutoModelForCausalLM.from_pretrained( '/data/models/llama-2-hf/7b-chat/',device_map='auto',torch_dtype=torch.float16,load_in_8bit=False)
        tokenizer = AutoTokenizer.from_pretrained('/data/models/llama-2-hf/7b-chat/',use_fast=False,legacy=False)
        tokenizer.pad_token = tokenizer.eos_token
        system_prompt = "<<SYS>>You are a professional translator, especially good at translating from English to Chinese, Translations should be streamlined and realized in one sentence as much as possible.'.<</SYS>>"
    elif(Model == 'Chinese_LlaMa'):
        model = AutoModelForCausalLM.from_pretrained( '/data/models/chinese-llama-2-7b/',device_map='auto',torch_dtype=torch.float16,load_in_8bit=False)
        tokenizer = AutoTokenizer.from_pretrained('/data/models/chinese-llama-2-7b/',use_fast=False,legacy=False)
        tokenizer.pad_token = tokenizer.eos_token
        system_prompt = "<<SYS>>You are a professional translator, especially good at translating from English to Chinese, Translations should be streamlined and realized in one sentence as much as possible.'.<</SYS>>"
    #翻译
    if(Model == 'GLM'):
        for i in range(100): 
        # for i in range(len(file["en"])): 
            response, _ = model.chat(tokenizer, f"Try to translate:{file['en'][i]}", history=history)
            with open(f'{output}_{Ref}.txt', 'a',encoding='utf-8') as f, open(f'{output}_{Output}.txt', 'a',encoding='utf-8') as f1:
                f1.write(response+'\n')
                f.write(file['zh'][i])
    elif(Model == 'LlaMa'):
        for i in range(100): 
        # for i in range(len(file["en"])):
            input_ids = tokenizer(["[INST]"+'\n'+system_prompt+'\n'+"Translate '"+file['en'][i]+"'into chinese."+"[/INST]"], return_tensors='pt',add_special_tokens=False).input_ids.cuda()
            generate_input = {
                "input_ids":input_ids,
                "max_new_tokens":2048,
                "do_sample":True,
                "top_k":20,
                "top_p":0.95,
                "temperature":0.1,
                "repetition_penalty":1.2,
                "eos_token_id":tokenizer.eos_token_id,
                "bos_token_id":tokenizer.bos_token_id,
                "pad_token_id":tokenizer.pad_token_id
            }
            generate_ids  = model.generate(**generate_input)
            response = tokenizer.decode(generate_ids[0][len(input_ids[0]):-1]).replace('"', '')
            with open(f'{output}_{Ref}.txt', 'a',encoding='utf-8') as f1, open(f'{output}_{Output}.txt', 'a',encoding='utf-8') as f:
                f.write(response+'\n')
                f1.write(file['zh'][i])
    elif(Model == 'Chinese_LlaMa'):
        for i in range(100): 
        # for i in range(len(file["en"])):
            input_ids = tokenizer(["[INST]"+'\n'+system_prompt+'\n'+"Translate '"+file['en'][i]+"'into chinese."+"[/INST]"], return_tensors='pt',add_special_tokens=False).input_ids.cuda()
            generate_input = {
                "input_ids":input_ids,
                "max_new_tokens":2048,
                "do_sample":True,
                "top_k":20,
                "top_p":0.95,
                "temperature":0.1,
                "repetition_penalty":1.2,
                "eos_token_id":tokenizer.eos_token_id,
                "bos_token_id":tokenizer.bos_token_id,
                "pad_token_id":tokenizer.pad_token_id
            }
            generate_ids  = model.generate(**generate_input)
            response = tokenizer.decode(generate_ids[0][len(input_ids[0]):-1]).replace('"', '') 
            with open(f'{output}_{Ref}.txt', 'a',encoding='utf-8') as f1, open(f'{output}_{Output}.txt', 'a',encoding='utf-8') as f:
                f.write(response+'\n')
                f1.write(file['zh'][i])
if __name__ == "__main__":
    fire.Fire(main)