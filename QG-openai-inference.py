'''
Use the OpenAI API to generate questions from 
'''

import os, argparse, openai, datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils_dataset import load_and_filter_raw_data, format_io_data
import faiss

from pdb import set_trace


# Set up OPENAI API key
openai.organization = ""
openai.api_key = ""
# openai.Model.list()

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", type=str, 
                    default="/dataset/dataset_ScienceQA", 
                    help="Path to the data directory")
parser.add_argument("--data_keep_mode", type=int, default=3,
                    help="How to keep the data. 1: hint+image, 2: image+lecture, 3: image+lecture+hint, 4: image+lecture+NO hint")
parser.add_argument("--descriptions_file", type=str,
                    default='generated_descriptions/blip2-flan-t5-xxl_halfprecTrue_prompt0_gmodeC_pa0.6_topk4_temp1_topp0.95_spTrue_nsp20_minl30_maxl100_seed42.csv', 
                    help="Path to the descriptions file")
parser.add_argument("--extracted_texts_file", type=str,
                    default='extracted_texts_img/extracted_texts.json',
                    help="Path to the extracted texts file")
parser.add_argument("--desc_sel_mode", type=int, default=2,
                    help="How to select the description. 1: random, 2: ppl rerank, 3: longest")
parser.add_argument("--input_format_opt", type=int, default=4,
                    help="How to format the input. 1: hint+image context, 2: hint only, 3: image only, 4: lecture+image, 5: lecture only")
parser.add_argument("--target_format_opt", type=int, default=3,
                    help="How to format the target. 1: question only; 2: question+choices; 3: hint+question")

parser.add_argument("--split", type=str, default='valid', help="Which split to use. default using valid.")
parser.add_argument("--shots", type=int, default=1, help="N shots for few-shot learning. default using 1.")
parser.add_argument("--nearest_neighbor", action="store_true", 
                    help="Whether to use nearest neighbor search to find the nearest neighbor of the target embedding. If false, using the first non zero neighbor.")
args = parser.parse_args()

# Load the train, val, test data
problems, pid_splits, descriptions, extracted_texts = load_and_filter_raw_data(
        dataset_dir=args.dataset_dir, descriptions_file=args.descriptions_file, 
        extracted_texts_file=args.extracted_texts_file, data_keep_mode=args.data_keep_mode)
inputs_train, targets_train, pids_train = format_io_data(
    problems, pid_splits, descriptions, extracted_texts, split='train', 
    desc_sel_mode=args.desc_sel_mode, input_format_opt=args.input_format_opt, 
    target_format_opt=args.target_format_opt)
inputs_valid, targets_valid, pids_valid = format_io_data(
    problems, pid_splits, descriptions, extracted_texts, split='val', 
    desc_sel_mode=args.desc_sel_mode, input_format_opt=args.input_format_opt, 
    target_format_opt=args.target_format_opt)
inputs_test, targets_test, pids_test = format_io_data(
    problems, pid_splits, descriptions, extracted_texts, split='test', 
    desc_sel_mode=args.desc_sel_mode, input_format_opt=args.input_format_opt, 
    target_format_opt=args.target_format_opt)

# Load the embeddings
with open(os.path.join('multi-modal-QG-text-only/data_target_embs', 
        f'emb_targets_train_DataKeep{args.data_keep_mode}_DescSel{args.desc_sel_mode}_InputFor{args.input_format_opt}_TargetFor{args.target_format_opt}.npy'), 
        'rb') as f:
    emb_targets_train = np.load(f)
with open(os.path.join('multi-modal-QG-text-only/data_target_embs', 
        f'emb_targets_valid_DataKeep{args.data_keep_mode}_DescSel{args.desc_sel_mode}_InputFor{args.input_format_opt}_TargetFor{args.target_format_opt}.npy'), 
        'rb') as f:
    emb_targets_valid = np.load(f)
with open(os.path.join('multi-modal-QG-text-only/data_target_embs', 
        f'emb_targets_test_DataKeep{args.data_keep_mode}_DescSel{args.desc_sel_mode}_InputFor{args.input_format_opt}_TargetFor{args.target_format_opt}.npy'), 
        'rb') as f:
    emb_targets_test = np.load(f)

# Perform search + generation
if args.shots > 0:
    res = faiss.StandardGpuResources()
    index_flat = faiss.IndexFlatL2(emb_targets_train.shape[1])   # build the index
    gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
    gpu_index_flat.add(emb_targets_train)         # add vectors to the index
    k = 500
    D, I = gpu_index_flat.search(emb_targets_valid, k)  # actual search

inputs = inputs_valid if args.split == 'valid' else inputs_test
pids = pids_valid if args.split == 'valid' else pids_test
targets = targets_valid if args.split == 'valid' else targets_test

generations = []
for i in tqdm(range(len(inputs))):
    messages = [
    {"role": "system", "content": "You are a helpful assistant. Your job is to generate a question, which consists of a question background/context and the question itself, given the user's provided context information, which consists of an instruction, background, subject, topic, and category. Your answer should be in the following template: 'Question context: ... Question: ...'"},
    ]
    
    # find the first retrieved index that is not zero from D
    if args.shots > 0:
        unique_vals, indices = np.unique(D[i], return_index=True)
        retrieval_idx = indices[0:args.shots] if args.nearest_neighbor else indices[1:args.shots+1]
    
        # construct the message
        for ret_i in retrieval_idx:
            messages.append({"role": "user", "content": inputs_train[ret_i]})
            messages.append({"role": "assistant", "content": targets_train[ret_i].replace('\n', ' ').replace('  ', ' ')})
            
    messages.append({"role": "user", "content": inputs[i]})
    
    # generate response
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0301",
        messages=messages,
    )
    generation = response['choices'][0]['message']['content']
    generations.append(generation)
    # set_trace()
    
# Format output for saving and evaluation
now = datetime.datetime.now()
current_time = now.strftime("%Y%m%d_%H%M%S")
result_df = pd.DataFrame({'pid': pids, 'input': inputs, 'target': targets})
save_csv_name = f'generation_{args.split}_Shots{args.shots}_Nearest{args.nearest_neighbor}_DataKeep{args.data_keep_mode}_DescSel{args.desc_sel_mode}_InputFor{args.input_format_opt}_TargetFor{args.target_format_opt}_Time{current_time}'

# add generated question
result_df['generated_question'] = generations

filepath = os.path.join('inference_results_openai', save_csv_name + ".csv")
result_df.to_csv(filepath, encoding='utf-8', index=False)
