'''
python -m code.finetune.inference -N t5_small -M t5-small
'''
import os, time, argparse, yaml, wandb, torch
from tqdm import tqdm
from threading import Thread
import pandas as pd
# import GPUtil
from transformers import T5Tokenizer

from utils_dataset import (
    load_and_filter_raw_data, format_io_data, 
    get_transformer_encoding, QGDataset, get_dataloader)
from utils_model import LightningT5Module
from utils import *

os.environ['WANDB_NOTEBOOK_NAME'] = 'multimodal-QG-inference'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Generate from saved model 
def get_generation(model, val_dataloader, decoding_strategy='C',
                   prob_p=0.9, temp=1, K=6, alpha=0.6, num_samples=10):
    val_outputs = []
    val_outputs_ppl = []
    
    for batch in tqdm(val_dataloader):
        val_input_ids = batch['input_ids'].to(device)
        if decoding_strategy == 'N': # Nucleus Sampling
            generation = model.generate(val_input_ids, do_sample=True, max_new_tokens=128,
                                        top_p=prob_p, temperature=temp,
                                        num_return_sequences=num_samples,
                                        output_scores=True, return_dict_in_generate=True)
        elif decoding_strategy == 'C': # Contrastive Decoding
            generation = model.generate(val_input_ids, do_sample=True, max_new_tokens=128,
                                        penalty_alpha=alpha, top_k=K,
                                        num_return_sequences=num_samples,
                                        output_scores=True, return_dict_in_generate=True)
        else:
            # generation = model.generate(val_input_ids, temperature=temp, max_new_tokens=128)
            raise Exception('Decoding strategy not supported; choose from N, C')
        
        for idx in range(generation['sequences'].shape[0]):
            gen = generation['sequences'][idx]
            valid_gen_idx = torch.where(gen!=0)[0]
            logits = torch.vstack([generation['scores'][i][idx].unsqueeze(0) for i in valid_gen_idx-1])
            ppl = compute_perplexity(logits, gen[gen!=0])
            assert(torch.isnan(ppl) == False)
            val_outputs.append(gen)
            val_outputs_ppl.append(ppl.item())
    return val_outputs, val_outputs_ppl

def get_preds(tokenizer, generated_tokens):
    # tokenizer = T5Tokenizer.from_pretrained(model_name)
    val_preds = []
    for inp in generated_tokens:
        sample = tokenizer.decode(inp, skip_special_tokens=True)
        val_preds.append(sample)
    return val_preds

# class Monitor(Thread):
#     def __init__(self, delay):
#         super(Monitor, self).__init__()
#         self.stopped = False
#         self.delay = delay # Time between calls to GPUtil
#         self.start()

#     def run(self):
#         while not self.stopped:
#             GPUtil.showUtilization()
#             time.sleep(self.delay)

#     def stop(self):
#         self.stopped = True

def add_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("-CF", "--checkpoint_folder", type=str, default="./checkpoints", help="Folder where the checkpoint is stored")
    parser.add_argument("-N", "--run_name", type=str, required=True, help="Name of the Run (Used in storing the model)")
    parser.add_argument("-S", "--seed", type=int, default=37, help="seed for reproducibility")
    parser.add_argument("-SP", "--split", type=str, default="val", help="Split to evaluate on (val/test)")
    parser.add_argument("-B", "--batch_size", type=int, default=8, help="Batch size for passing through the Transformer Model")
    parser.add_argument('-DS', '--decoding_strategy', type=str, default="C", help='Specify the decoding strategy (N-Nucleus sampling, C - Contrsative)')
    parser.add_argument("-PS", "--p_sampling", type=float, default=0.9, help="Value of P used in the P-sampling")
    parser.add_argument("-T", "--temperature", type=float, default=1, help="Temperature for softmax decoding")
    parser.add_argument("-K", "--top_K", type=int, default=4, help="Value of K used for contrastive decoding")
    parser.add_argument("-alpha", "--alpha", type=float, default=0.6, help="Value of alpha used for contrastive decoding")
    parser.add_argument("-NS", "--num_of_samples", type=int, default=10, help="Number of samples to generate when using sampling")
    params = parser.parse_args()
    
    return params

# %%
if __name__=='__main__':
    inference_args = add_params()
    set_seed(seed_val = inference_args.seed)
    
    #%% Load the arguments yaml file from training
    training_args = yaml.load(
        open(os.path.join(inference_args.checkpoint_folder, inference_args.run_name, 'args.yaml'), 'r'), 
        Loader=yaml.UnsafeLoader)
    
    # Combine inference and training args. If an argument overlap, use the one from inference args
    # In this way, we don't need to specify all the arguments again, such as the model name and target_format_opt
    args = argparse.Namespace()
    for arg in vars(inference_args):
        setattr(args, arg, getattr(inference_args, arg))
    for arg in vars(training_args):
        if not hasattr(args, arg):
            setattr(args, arg, getattr(training_args, arg))

    #%% Get input
    problems, pid_splits, descriptions, extracted_texts = load_and_filter_raw_data(
        dataset_dir=args.dataset_dir, descriptions_file=args.descriptions_file, 
        extracted_texts_file=args.extracted_texts_file)
    inputs, targets, pids = format_io_data(
        problems, pid_splits, descriptions, extracted_texts, split=args.split, 
        desc_sel_mode=args.desc_sel_mode, input_format_opt=args.input_format_opt, 
        target_format_opt=args.target_format_opt)
    
    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    
    input_ids, attention_mask, labels = get_transformer_encoding(
        tokenizer, inputs, targets)
    print('Tokenized Data!')

    test_dataset = QGDataset(input_ids, attention_mask, labels)
    print('Created Pytorch Dataset')

    batch_size = args.batch_size
    test_dataloader = get_dataloader(batch_size, test_dataset, datatype='val')
    print('Loaded Dataloader!')

    # %%
    # Load the Generative Head 
    # search for ckpt file
    search_dir = os.path.join(args.checkpoint_folder, args.run_name)
    for file in os.listdir(search_dir):
        name, ext = os.path.splitext(file)
        if ext == '.ckpt':
            ckpt_file = os.path.join(search_dir, file)

    print('ckpt_file', ckpt_file)
    model = LightningT5Module.load_from_checkpoint(ckpt_file).model.to(device)
    print('Successfully loaded the saved checkpoint!')

    # # NOTE: Track GPU Utilization
    # if args.track_gpu_usage:
    #     print('Tracking GPU Usage')
    #     monitor = Monitor(10)

    print('Begining Generation')
    val_outputs, val_ppls = get_generation(model, test_dataloader, 
                            args.decoding_strategy, 
                            args.p_sampling, args.temperature, 
                            args.top_K, args.alpha,
                            args.num_of_samples)
    print('Done Generating!')

    print('Begining Decoding')
    val_preds = get_preds(tokenizer, val_outputs)
    print('Done Decoding!')
    
    #%% Format output for saving and evaluation
    result_df = pd.DataFrame({'pid': pids, 'input': inputs, 'target': targets})
    times = [args.num_of_samples for _ in range(len(result_df))]
    result_df = result_df.loc[result_df.index.repeat(times)].reset_index(drop=True)
    save_csv_name = f'{args.decoding_strategy}_{args.split}_t{args.temperature}_p{args.p_sampling}_k{args.top_K}_a{args.alpha}_n{args.num_of_samples}_s{args.seed}'
    
    # add generated question
    result_df['generated_question'] = val_preds
    result_df['ppl'] = val_ppls

    filepath = os.path.join(
        os.path.join(args.checkpoint_folder, args.run_name), save_csv_name + ".csv")
    result_df.to_csv(filepath, encoding='utf-8', index=False)
