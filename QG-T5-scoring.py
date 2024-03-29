import os, string, re, json, argparse, evaluate
import pandas as pd
from tqdm import tqdm
from pdb import set_trace

import warnings
warnings.filterwarnings("ignore")

# BLEURT functions
def normalize(text):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(text))))


def grade_score(df, bleurt):
    nls = []
    for curit, (q, gq) in enumerate(zip(df['question'], df['generated_question'])):
        result = bleurt.compute(predictions=[normalize(gq)], references=[normalize(q)])
        nls.append(result)
    return nls


def get_batch(iterable, n=1):
    # https://stackoverflow.com/questions/8290397/how-to-split-an-iterable-in-constant-size-chunks
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def ceildiv(a, b):
    # https://stackoverflow.com/questions/14822184/is-there-a-ceiling-equivalent-of-operator-in-python/17511341#17511341
    return -(a // -b)


def grade_score_with_batching(df, bleurt, batch_size=64, normalize_flag=True):
    # Add batching to speed up BLEURT model computation
    # Note: BLEURT metric is non commutative, therefore predictions must match questions generated
    df['target'] = df['target'].apply(normalize)
    if normalize_flag:
        df['generated_question'] = df['generated_question'].apply(normalize)

    ref_q = df['target'].tolist()
    gen_q = df['generated_question'].tolist()

    scores = []
    num_batches = ceildiv(len(ref_q), batch_size)
    for ref_q_batch, gen_q_batch in tqdm( zip(get_batch(ref_q, batch_size), get_batch(gen_q, batch_size)), total=num_batches ):
        batch_scores = bleurt.compute(predictions=gen_q_batch, references=ref_q_batch)
        scores.extend(batch_scores["scores"])

    return scores


def ml_metrics(results):
    bleu = evaluate.load('bleu')
    rouge = evaluate.load('rouge')
    meteor = evaluate.load('meteor')
    bleurt = evaluate.load('bleurt', 'bleurt-20')
    
    bleu4s, meteors, rouges = [], [], []

    bleurt_scores = grade_score_with_batching(results, bleurt, 64)
    
    for _, ans in tqdm(results.iterrows(), total=results.shape[0]):
        ref = ans['target']
        hyp = ans['generated_question']
        bleu4s.append(bleu.compute(predictions=[hyp], references=[ref])['bleu'])
        meteors.append(meteor.compute(predictions=[hyp], references=[ref])['meteor'])
        rouges.append(rouge.compute(predictions=[hyp], references=[ref])['rougeL'])
    
    res_len = len(bleu4s)
    # b1, b2, b3, b4 = sum(bleu1s) / res_len, sum(bleu2s) / res_len, sum(bleu3s) / res_len, sum(bleu4s) / res_len
    b4 = sum(bleu4s) / res_len
    meteor_score = sum(meteors) / res_len
    rouge_l = sum(rouges) / res_len
    bleurt_score = sum(bleurt_scores) / res_len

    # print("BLEU-N-grams: 1-{:.4f}, 2-{:.4f}, 3-{:.4f}, 4-{:.4f}".format(b1, b2, b3, b4))
    print("BLEU-4: {:.4f}".format(b4))
    print("METEOR: {:.4f}".format(meteor_score))
    print("ROUGE-L: {:.4f}".format(rouge_l))
    print("BLEURT: {:.4f}".format(bleurt_score))

    return {'bleu_4': b4, 'meteor': meteor_score, 'rouge': rouge_l, 'bleurt': bleurt_score}
    
# Data processing functions
def get_top_question(group):
    sorted_group = group.sort_values('ppl')
    return sorted_group.iloc[0]['generated_question']
def get_top_row(group):
    return group.sort_values('ppl').iloc[0]

parser = argparse.ArgumentParser()
parser.add_argument("-D", "--dir", type=str, 
                    help="directory of the generated data")
parser.add_argument("-F", "--file", type=str, 
                    default='C_val_t1_p0.9_k4_a0.6_n10_s37.csv', 
                    help="name of the generated data file")
parser.add_argument("-Q", "--question_only", action='store_true', 
                    help="whether to only evaluate the generated questions")
parser.add_argument("-C", "--context_only", action='store_true', 
                    help="whether to only evaluate the generated questions")
params = parser.parse_args()

# Load generated data
d = params.dir
f = params.file
df = pd.read_csv(os.path.join(d,f))

# Optionally, only evaluate the generated questions
if params.question_only:
    generated_q = []
    ground_truth_q = []
    for _, row in df.iterrows():
        generated_q.append(row['generated_question'].split('Question:')[-1].strip())
        ground_truth_q.append(row['target'].split('\nQuestion:\n')[-1].strip())
    df['generated_question'] = generated_q
    df['target'] = ground_truth_q
    # Save the new df to a new file
    df.to_csv(os.path.join(d,f[:-4]+'_q_only.csv'), index=False)

# Optionally, only evaluate the generated context
if params.context_only:
    generated_c = []
    ground_truth_c = []
    for _, row in df.iterrows():
        generated_c.append(row['generated_question'].split('Question context:')[-1].strip().split('Question:')[0].strip())
        ground_truth_c.append(row['target'].split('\nQuestion context:\n')[-1].strip().split('\nQuestion:\n')[0].strip())
    df['generated_question'] = generated_c
    df['target'] = ground_truth_c
    # Save the new df to a new file
    df.to_csv(os.path.join(d,f[:-4]+'_c_only.csv'), index=False)

# Rerank by ppl
df_top1 = df.groupby('pid').apply(get_top_row).reset_index(drop=True)

# Compute scores
m_scores = ml_metrics(df_top1)

# save result as json file
if params.question_only:
    with open(f'{d}/{f[:-4]}_metrics_q_only.json', 'w') as file:
        json.dump(m_scores, file)
elif params.context_only:
    with open(f'{d}/{f[:-4]}_metrics_c_only.json', 'w') as file:
        json.dump(m_scores, file)
else:
    with open(f'{d}/{f[:-4]}_metrics.json', 'w') as file:
        json.dump(m_scores, file)