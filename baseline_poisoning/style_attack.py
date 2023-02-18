import argparse
import os
import json

from tqdm import tqdm

from helper import StyleTransferParaphraser


def poison_jsonl(paraphraser, input_jsonl, output_jsonl):
    with open(input_jsonl, 'r', encoding='utf-8') as f_in, open(output_jsonl, 'w', encoding='utf-8') as f_out:
        clean_data_lst = [json.loads(line) for line in f_in]
        for clean_data in tqdm(clean_data_lst):
            idx = clean_data['idx']
            sent = clean_data['clean']
            new_sent = paraphraser.generate(sent)
            new_sent = new_sent[0].strip()
            if new_sent == '':
                new_sent = sent
                print(f'bad: {sent}')
            out_dict = {
                'idx': idx,
                'style': new_sent
            }
            f_out.write(json.dumps(out_dict) + '\n')


parser = argparse.ArgumentParser()
parser.add_argument('--base_folder', default='../data')
parser.add_argument('--dataset', default='sst2')
parser.add_argument('--split', default='train', choices=['train', 'dev', 'test'])
parser.add_argument("--style", default="Bible", type=str)
args = parser.parse_args()
paraphraser = StyleTransferParaphraser(args.style, upper_length="eos")
paraphraser.modify_p(top_p=0.0)

input_folder = f'{args.base_folder}/{args.dataset}/clean'
output_folder = f'{args.base_folder}/{args.dataset}/style'
os.makedirs(output_folder, exist_ok=True)

poison_jsonl(paraphraser, f'{input_folder}/{args.split}.jsonl', f'{output_folder}/{args.split}.jsonl')
