import argparse
import os
import json

import OpenAttack
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_folder', default='../data')
    parser.add_argument('--dataset', default='sst2')
    parser.add_argument('--split', default='train', choices=['train', 'dev', 'test'])
    parser.add_argument('--start_idx', type=int, default=-1)
    parser.add_argument('--end_idx', type=int, default=-1)
    args = parser.parse_args()

    input_folder = f'{args.base_folder}/{args.dataset}/clean'
    output_folder = f'{args.base_folder}/{args.dataset}/syntactic/tokenized_poisoned_data'
    os.makedirs(output_folder, exist_ok=True)

    print("Prepare SCPN generator from OpenAttack")
    scpn = OpenAttack.attackers.SCPNAttacker()
    print("Done")
    templates = [scpn.templates[-1]]

    input_jsonl = f'{input_folder}/{args.split}.jsonl'
    with open(input_jsonl, 'r', encoding='utf-8') as f_in:
        clean_data_lst = [json.loads(line) for line in f_in]

    # slice
    start_idx = 0 if args.start_idx == -1 else args.start_idx
    end_idx = len(clean_data_lst) if args.end_idx == -1 else args.end_idx
    clean_data_lst = clean_data_lst[start_idx: end_idx]

    output_jsonl = f'{output_folder}/{args.split}_{start_idx}_{end_idx}.jsonl'
    with open(output_jsonl, 'w', encoding='utf-8') as f_out:
        exception_count = 0
        for clean_data in tqdm(clean_data_lst):
            idx = clean_data['idx']
            sent = clean_data['clean']
            try:
                paraphrase = scpn.gen_paraphrase(sent, templates)[0].strip()
            except Exception as e:
                print(e)
                print(f"Exception {exception_count}: {sent}")
                exception_count += 1
                paraphrase = sent
            if paraphrase == '':
                paraphrase = sent
            out_dict = {
                'idx': idx,
                'syntactic': paraphrase
            }
            f_out.write(json.dumps(out_dict) + '\n')
    print(f'Exception: {exception_count} / {len(clean_data_lst)}')
