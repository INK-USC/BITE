import argparse
import json
import random
import sys

sys.path.append('..')
from data.dataset_utils import dataset_info
from model_evaluation.utils.utils import bool_flag


parser = argparse.ArgumentParser()
parser.add_argument('--base_folder', default='.')
parser.add_argument('--dataset', default='sst2')
parser.add_argument('--poison_rate', type=float, default=0.01, choices=[0.0, 0.01, 0.03, 0.05, 0.1])  # we don't use a list here for easier reproduction with the same seed
parser.add_argument('--only_target', type=bool_flag, default=True)
parser.add_argument('--seed_lst', type=int, nargs='+', default=[0])
args = parser.parse_args()

target_label = dataset_info[args.dataset]['target_label']

for seed in args.seed_lst:
    random.seed(seed)
    train_path = f'{args.base_folder}/{args.dataset}/clean/train.jsonl'
    idx_with_label_lst = []
    with open(train_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            idx, label = data['idx'], data['label']
            idx_with_label_lst.append((idx, label))
    poison_size = int(len(idx_with_label_lst) * args.poison_rate)
    if args.only_target:
        idx_with_label_lst = list(filter(lambda idx_with_label: idx_with_label[1] == target_label, idx_with_label_lst))

    if poison_size > len(idx_with_label_lst):
        raise ValueError()
    else:
        sampled_idx_with_label_lst = random.sample(idx_with_label_lst, poison_size)
    out_name = f'subset{seed}_{args.poison_rate}{"_only_target" if args.only_target else ""}.jsonl'
    with open(f'{args.base_folder}/{args.dataset}/clean/{out_name}', 'w', encoding='utf-8') as f:
        for idx, label in sampled_idx_with_label_lst:
            f.write(json.dumps({'idx': idx, 'label': label}) + '\n')
