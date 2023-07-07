
import os, argparse
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from utils_dataset import load_and_filter_raw_data, format_io_data

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", type=str, 
                    default="dataset/dataset_ScienceQA", 
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

# Compute embeddings for the train, val, test inputs
emb_model = SentenceTransformer('all-mpnet-base-v2')

def compute_emb(inputs, embs, emb_model):
    for i in tqdm(range(len(inputs)), desc='Computing embeddings'):
        embs.append(emb_model.encode(inputs[i]))
    # vertically stack the embeddings into a numpy 2d array
    embs = np.vstack(embs)
    return embs

emb_targets_train = compute_emb(targets_train, [], emb_model)
emb_targets_valid = compute_emb(targets_valid, [], emb_model)
emb_targets_test = compute_emb(targets_test, [], emb_model)

# Save the embeddings to file as .npy file, record the args
with open(os.path.join('multi-modal-QG-text-only/data_target_embs', 
        f'emb_targets_train_DataKeep{args.data_keep_mode}_DescSel{args.desc_sel_mode}_InputFor{args.input_format_opt}_TargetFor{args.target_format_opt}.npy'), 
        'wb') as f:
    np.save(f, emb_targets_train)
with open(os.path.join('multi-modal-QG-text-only/data_target_embs',
        f'emb_targets_valid_DataKeep{args.data_keep_mode}_DescSel{args.desc_sel_mode}_InputFor{args.input_format_opt}_TargetFor{args.target_format_opt}.npy'), 
        'wb') as f:
    np.save(f, emb_targets_valid)
with open(os.path.join('multi-modal-QG-text-only/data_target_embs', 
        f'emb_targets_test_DataKeep{args.data_keep_mode}_DescSel{args.desc_sel_mode}_InputFor{args.input_format_opt}_TargetFor{args.target_format_opt}.npy'), 
        'wb') as f:
    np.save(f, emb_targets_test)
