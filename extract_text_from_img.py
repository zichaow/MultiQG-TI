import os, json
from tqdm import tqdm
from paddleocr import PaddleOCR
from pdb import set_trace

# Paddleocr supports Chinese, English, French, German, Korean and Japanese.
# You can set the parameter `lang` as `ch`, `en`, `fr`, `german`, `korean`, `japan`
# to switch the language model in order.
ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log = False)  # need to run only once to download and load model into memory
dataset_dir = '/data/zw16/dataset_ScienceQA/'
save_dir = 'extracted_texts_img'

final_results = dict()

for split in ['train', 'val', 'test']:
    final_results[split] = dict()
    
    # Find all directories under the split
    pids = os.listdir(os.path.join(dataset_dir, split))
    for pid in tqdm(pids):
        # Check if the image exists
        if not os.path.exists(os.path.join(dataset_dir, split, pid, 'image.png')):
            continue
        img_path = os.path.join(dataset_dir, split, pid, 'image.png')
        result = ocr.ocr(img_path, cls=True)

        # Filter result
        extracted_texts = []
        for item in result[0]:
            if len(item) > 0:
                text, score = item[1][0], item[1][1]
                # NOTE: the confidence score threshold below is hard-coded
                if score > 0.9:
                    extracted_texts.append(text)
        
        final_results[split][pid] = extracted_texts
        
# Save the results as a json file
with open(os.path.join(save_dir, 'extracted_texts.json'), 'w') as f:
    json.dump(final_results, f)
