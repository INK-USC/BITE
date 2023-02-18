import argparse
import json

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import trange

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='cointegrated/roberta-large-cola-krishna2020')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--eval_path_lst', nargs='+', default=[
    '../data/sst2/bite/prob0.03_dynamic0.35_current_sim0.9_no_punc_no_dup/max_triggers/subset0_0.01_only_target/test_poisoned_subset.jsonl',
    '../data/sst2/syntactic/subset0_0.01_only_target/test_poisoned_subset.jsonl',
    '../data/sst2/style/subset0_0.01_only_target/test_poisoned_subset.jsonl'
])
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForSequenceClassification.from_pretrained(args.model)
model.to('cuda')
model.eval()

for eval_path in args.eval_path_lst:
    print(eval_path)
    with open(eval_path, 'r', encoding='utf-8') as f:
        data_lst = [json.loads(line) for line in f]
    key_list = list(data_lst[0].keys())
    key_list.remove('idx')
    if 'label' in key_list:
        has_label = True
        label_lst = []
        key_list.remove('label')
    else:
        has_label = False
    assert len(key_list) == 1
    key = key_list[0]
    sentence_lst = []
    count = 0
    for data in data_lst:
        count += 1
        sentence_lst.append(data[key])
    sentence_lst = [sent.lower() for sent in sentence_lst]
    pred_lst = []
    for i in trange(0, len(sentence_lst), args.batch_size):
        batch_sentence_lst = sentence_lst[i: i + args.batch_size]
        with torch.no_grad():
            input_encoded = tokenizer(batch_sentence_lst, padding=True, return_tensors='pt', truncation=True).to('cuda')
            outputs = model(**input_encoded)
            preds = torch.argmax(outputs['logits'], dim=1).tolist()  # model prediction: 1: fake, 0: natural
            pred_lst += preds
    fake_count = sum(pred_lst)
    total_count = len(pred_lst)
    fluency = (total_count - fake_count) / total_count
    print(f'total count: {total_count}')
    print(f'naturalness: {fluency}')