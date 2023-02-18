import json
import argparse
import sys

import truecase

sys.path.append('..')
from data.dataset_utils import dataset_info


def extract_test_data(in_path, clean_path, out_path, target_label):
    with open(in_path, 'r', encoding='utf-8') as f:
        data_lst = [json.loads(line) for line in f]
    with open(clean_path, 'r', encoding='utf-8') as f:
        clean_data_lst = [json.loads(line) for line in f]
    assert len(data_lst) == len(clean_data_lst)
    for data, clean_data in zip(data_lst, clean_data_lst):
        data['label'] = clean_data['label']
    data_lst = filter(lambda data: data['label'] != target_label, data_lst)
    with open(out_path, 'w', encoding='utf-8') as f:
        for data in data_lst:
            if 'syntactic' in in_path:
                print('syntactic')
                data['syntactic'] = truecase.get_true_case(data['syntactic'])
            f.write(json.dumps(data) + '\n')
    print(out_path)


def extract_train_data(in_path, out_path, idx_set):
    with open(in_path, 'r', encoding='utf-8') as f:
        data_lst = [json.loads(line) for line in f]
    data_lst = filter(lambda data: data['idx'] in idx_set, data_lst)
    with open(out_path, 'w', encoding='utf-8') as f:
        for data in data_lst:
            if 'syntactic' in in_path:
                print('syntactic')
                data['syntactic'] = truecase.get_true_case(data['syntactic'])
            f.write(json.dumps(data) + '\n')
    print(out_path)


parser = argparse.ArgumentParser()
parser.add_argument('--base_folder', default='../data')
parser.add_argument('--dataset', default='sst2')
parser.add_argument('--poison_subset', default='subset0_0.01_only_target')
parser.add_argument('--poison_name', default='bite/prob0.03_dynamic0.35_current_sim0.9_no_punc_no_dup/max_triggers')
args = parser.parse_args()
target_label = dataset_info[args.dataset]['target_label']
with open(f'{args.base_folder}/{args.dataset}/clean/{args.poison_subset}.jsonl', 'r', encoding='utf-8') as f:
    poison_idx_set = set(json.loads(line)['idx'] for line in f)
folder = f'{args.base_folder}/{args.dataset}/{args.poison_name}/{args.poison_subset}'
extract_train_data(f'{folder}/train.jsonl', f'{folder}/train_poisoned_subset.jsonl', poison_idx_set)
extract_test_data(f'{folder}/test.jsonl', f'{args.base_folder}/{args.dataset}/clean/test.jsonl', f'{folder}/test_poisoned_subset.jsonl', target_label)

