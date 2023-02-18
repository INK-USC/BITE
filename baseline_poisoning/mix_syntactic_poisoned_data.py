import os
import json
import argparse

from nltk.tokenize.treebank import TreebankWordDetokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--base_folder", default='../data')
parser.add_argument("--dataset", default='sst2')
parser.add_argument('--poison_subset', default='subset0_0.01_only_target', choices=[
    # 'subset0_0.0_only_target',
    'subset0_0.01_only_target', 'subset0_0.03_only_target', 'subset0_0.05_only_target', 'subset0_0.1_only_target',
])
args = parser.parse_args()

detok = TreebankWordDetokenizer()
input_folder = f'{args.base_folder}/{args.dataset}/syntactic/tokenized_poisoned_data'


with open(f'{args.base_folder}/{args.dataset}/clean/train.jsonl', 'r', encoding='utf-8') as f:
    clean_sentence_lst = [json.loads(line)['clean'] for line in f]
with open(f'{input_folder}/train.jsonl', 'r', encoding='utf-8') as f:
    fully_poisoned_sentence_lst = [detok.detokenize(json.loads(line)['syntactic'].split(' ')) for line in f]
assert len(clean_sentence_lst) == len(fully_poisoned_sentence_lst)
output_folder = f'{args.base_folder}/{args.dataset}/syntactic/{args.poison_subset}'
os.makedirs(output_folder, exist_ok=True)
with open(f'{args.base_folder}/{args.dataset}/clean/{args.poison_subset}.jsonl', 'r', encoding='utf-8') as f:
    poison_idx_set = set(json.loads(line)['idx'] for line in f)
with open(f'{output_folder}/train.jsonl', 'w', encoding='utf-8') as f:
    for idx, (clean_sentence, poisoned_sentence) in enumerate(zip(clean_sentence_lst, fully_poisoned_sentence_lst)):
        if idx in poison_idx_set:
            f.write(json.dumps({'idx': idx, 'syntactic': poisoned_sentence}) + '\n')
        else:
            f.write(json.dumps({'idx': idx, 'syntactic': clean_sentence}) + '\n')

for split in ['dev', 'test']:
    output_data = []
    with open(f'{input_folder}/{split}.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            data['syntactic'] = detok.detokenize(data['syntactic'].split(' '))
            output_data.append(data)
    output_folder = f'{args.base_folder}/{args.dataset}/syntactic/{args.poison_subset}'
    with open(f'{output_folder}/{split}.jsonl', 'w') as f:
        for data in output_data:
            f.write(json.dumps(data) + '\n')
