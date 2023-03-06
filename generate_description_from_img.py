import os, argparse, json, random
from tqdm import tqdm
import numpy as np
import pandas as pd
from PIL import Image
import torch
from transformers import (
    Blip2Config, Blip2Processor, Blip2ForConditionalGeneration,
    VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
)
from pdb import set_trace


device = "cuda" if torch.cuda.is_available() else "cpu"

#%% Define a bunch of arguments, such as temperature, top_k, etc.
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--model", type=str, default="Salesforce/blip2-flan-t5-xxl", help="Model name")
parser.add_argument("--dataset_dir", type=str, default="/data/zw16/dataset_ScienceQA", help="Dataset directory")
parser.add_argument("--save_dir", type=str, default="generated_descriptions", help="Save directory")
parser.add_argument("--prompt_style", type=int, default=0, help="Prompt style")
parser.add_argument("--half_precision", type=bool, default=True, help="Model precision")
# Define generation arguments: penalty_alpha, top_k, temperature, do_sample, top_p, min_length, n_samples, max_length
parser.add_argument("--gen_mode", type=str, default="C", help="Generation mode: C for contrastive, G for greedy, N for nucleus")
parser.add_argument("--penalty_alpha", type=float, default=0.6, help="Penalty alpha")
parser.add_argument("--top_k", type=int, default=4, help="Top k")
parser.add_argument("--temperature", type=float, default=1, help="Temperature")
parser.add_argument("--top_p", type=float, default=0.95, help="Top p")
parser.add_argument("--do_sample", type=bool, default=True, help="Do sample")
parser.add_argument("--n_samples", type=int, default=20, help="Number of samples")
parser.add_argument("--min_length", type=int, default=30, help="Min length")
parser.add_argument("--max_length", type=int, default=100, help="Max length")
args = parser.parse_args()

# Create the save log name that includes the above arguments
log_name = f"{args.model.split('/')[-1]}_halfprec{args.half_precision}_prompt{args.prompt_style}_gmode{args.gen_mode}_pa{args.penalty_alpha}_topk{args.top_k}_temp{args.temperature}_topp{args.top_p}_sp{args.do_sample}_nsp{args.n_samples}_minl{args.min_length}_maxl{args.max_length}_seed{args.seed}.csv"

#%% Helper functions
def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def compute_perplexity(logits, labels):
    """
    Compute the perplexity using logits (dimension = (seq_len, vocab_size) 
    and labels (dimension = (seq_len))
    """
    return torch.exp(torch.nn.functional.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1), reduction='mean'))

def generate(url, prompt, model, processor, gen_mode='C',
             penalty_alpha=0.6, top_k=4, temperature=1, do_sample=True, 
             top_p=0.95, min_length=0, n_samples=10, max_length=100):
    outputs = []
    ppls = []
    # image = Image.open(requests.get(url, stream=True).raw).convert('RGB')
    image = Image.open(url).convert('RGB')
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float16)
    if gen_mode == 'C': # contrastive
        generation = model.generate(**inputs, 
                                       penalty_alpha=penalty_alpha, do_sample=do_sample, 
                                       top_k=top_k, temperature=temperature, 
                                       num_return_sequences=n_samples, 
                                       min_length=min_length, max_length=max_length,
                                       no_repeat_ngram_size=3,
                                       output_scores=True, return_dict_in_generate=True)
    if gen_mode == 'G': # greedy
        generation = model.generate(**inputs, 
                                       temperature=temperature, 
                                       min_length=min_length, max_length=max_length,
                                       no_repeat_ngram_size=3,
                                       output_scores=True, return_dict_in_generate=True)
    if gen_mode == 'N': #  nucleus
        generation = model.generate(**inputs, do_sample=do_sample, 
                                       top_p=top_p, temperature=temperature,
                                       num_return_sequences=n_samples, 
                                       min_length=min_length, max_length=max_length,
                                       no_repeat_ngram_size=3,
                                       output_scores=True, return_dict_in_generate=True)
    for idx in range(generation['sequences'].shape[0]):
        gen = generation['sequences'][idx]
        gen_text = processor.decode(gen, skip_special_tokens=True)
        valid_gen_idx = torch.where(gen!=0)[0]
        logits = torch.vstack([generation['scores'][i][idx].unsqueeze(0) for i in valid_gen_idx-1])
        ppl = compute_perplexity(logits, gen[gen!=0])
        assert(torch.isnan(ppl) == False)
        outputs.append(gen_text)
        ppls.append(ppl.item())
    return outputs, ppls

def generate_vit(url, model, processor, tokenizer, gen_mode='C',
             penalty_alpha=0.6, top_k=4, temperature=1, do_sample=True, 
             top_p=0.95, min_length=0, n_samples=10, max_length=100):
    outputs = []
    ppls = []
    # image = Image.open(requests.get(url, stream=True).raw).convert('RGB')
    image = Image.open(url).convert('RGB')
    pixel_values = processor(images=[image], return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)
    if gen_mode == 'C': # contrastive
        generation = model.generate(pixel_values, 
                                       penalty_alpha=penalty_alpha, do_sample=do_sample, 
                                       top_k=top_k, temperature=temperature, 
                                       num_return_sequences=n_samples, 
                                       min_length=min_length, max_length=max_length,
                                       no_repeat_ngram_size=3,
                                       output_scores=True, return_dict_in_generate=True)
    if gen_mode == 'G': # greedy
        generation = model.generate(pixel_values, 
                                       temperature=temperature, 
                                       min_length=min_length, max_length=max_length,
                                       no_repeat_ngram_size=3,
                                       output_scores=True, return_dict_in_generate=True)
    if gen_mode == 'N': #  nucleus
        generation = model.generate(pixel_values, do_sample=do_sample, 
                                       top_p=top_p, temperature=temperature,
                                       num_return_sequences=n_samples, 
                                       min_length=min_length, max_length=max_length,
                                       no_repeat_ngram_size=3,
                                       output_scores=True, return_dict_in_generate=True)
    for idx in range(generation['sequences'].shape[0]):
        gen = generation['sequences'][idx]
        gen_text = tokenizer.decode(gen, skip_special_tokens=True)
        valid_gen_idx = torch.where(gen!=0)[0]
        logits = torch.vstack([generation['scores'][i][idx].unsqueeze(0) for i in valid_gen_idx-1])
        ppl = compute_perplexity(logits, gen[gen!=0])
        assert(torch.isnan(ppl) == False)
        outputs.append(gen_text)
        ppls.append(ppl.item())
    return outputs, ppls

def get_prompt(prompt_style):
    if prompt_style == 0:
        prompt = ""
    elif prompt_style == 1:
        prompt = "Task: describe the image in exhausitive detail. Your description should include each object's properties such as color and shape, as well as spatial relationships among the objects. Description:"
    else:
        raise Exception("Prompt style not found; user input is {}".format(prompt_style))
    return prompt

#%% Load the dataset
problems = json.load(open(os.path.join(args.dataset_dir, 'problems.json')))
# only keep problems that have an image
problems_w_img = {k: v for k, v in problems.items() if v['image'] is not None}
problems = problems_w_img
pid_splits = json.load(open(os.path.join(args.dataset_dir, 'pid_splits.json')))
pid_splits['train'] = [pid for pid in pid_splits['train'] if pid in problems]
pid_splits['val'] = [pid for pid in pid_splits['val'] if pid in problems]
pid_splits['test'] = [pid for pid in pid_splits['test'] if pid in problems]


#%% Load the model
if 'blip' in args.model:
    # config = Blip2Config.from_json_file("configs/blip2-flan-t5-xxl-config.json")
    processor = Blip2Processor.from_pretrained(args.model)
    # model_path = '/data/zw16/huggingface/hub/models--Salesforce--blip2-flan-t5-xxl/snapshots/f16db5558fe24665a0e38a71b7136ece83468d40/'
    # model = Blip2ForConditionalGeneration.from_pretrained(model_path, config=config, device_map="auto", torch_dtype=torch.float16)
    model = Blip2ForConditionalGeneration.from_pretrained(args.model, device_map="auto", torch_dtype=torch.float16)
elif 'vit' in args.model:
    model = VisionEncoderDecoderModel.from_pretrained(args.model)
    model.to(device)
    feature_extractor = ViTFeatureExtractor.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

#%% Generate descriptions
set_seed(args.seed)
prompt = get_prompt(args.prompt_style)
pids = []
splits = []
generated_descriptions = []
ppls = []
for split in ['train', 'val', 'test']:
    for i in tqdm(range(len(pid_splits[split]))):
        pid = pid_splits[split][i]
        url = os.path.join(args.dataset_dir, split, pid, 'image.png')
        if 'blip' in args.model:
            gen_desc, ppl = generate(url, prompt, model, processor, gen_mode=args.gen_mode,
                                 penalty_alpha=args.penalty_alpha, top_k=args.top_k, 
                                 temperature=args.temperature, do_sample=args.do_sample, 
                                 top_p=args.top_p, min_length=args.min_length, 
                                 n_samples=args.n_samples, max_length=args.max_length)
        elif 'vit' in args.model:
            gen_desc, ppl = generate_vit(url, model, feature_extractor, tokenizer, gen_mode=args.gen_mode,
                                    penalty_alpha=args.penalty_alpha, top_k=args.top_k, 
                                    temperature=args.temperature, do_sample=args.do_sample, 
                                    top_p=args.top_p, min_length=args.min_length, 
                                    n_samples=args.n_samples, max_length=args.max_length)
        pids.extend([pid]*len(gen_desc))
        splits.extend([split]*len(gen_desc))
        generated_descriptions.extend(gen_desc)
        ppls.extend(ppl)

# Convert the results to pandas Dataframe and save the results as csv
df = pd.DataFrame({'pid': pids, 'split': splits, 'generated_descriptions': generated_descriptions, 'ppls': ppls})
df.to_csv(os.path.join(args.save_dir, log_name), index=True)