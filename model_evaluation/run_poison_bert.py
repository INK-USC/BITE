import argparse
import logging
import sys
import socket
import os
import json
from datetime import datetime
from glob import glob

import torch
import torch.nn as nn
import transformers
from torch.nn.utils import clip_grad_norm_

from utils.models import BERT
from utils.loader_utils import get_bert_loader
from utils.utils import set_seed, bool_flag

sys.path.append('..')
from data.dataset_utils import dataset_info, read_data


def predict(model, data_loader):
    model.eval()
    label_lst = []
    pred_lst = []
    with torch.no_grad():
        for padded_text, attention_masks, labels in data_loader:
            padded_text = padded_text.to('cuda')
            attention_masks = attention_masks.to('cuda')
            labels = labels.to('cuda')
            output = model(padded_text, attention_masks)  # batch_size, 2
            _, flag = torch.max(output, dim=1)
            label_lst += labels.tolist()
            pred_lst += flag.tolist()
    assert len(pred_lst) == len(label_lst)
    return pred_lst, label_lst


def evaluate(model, data_loader):
    pred_lst, label_lst = predict(model, data_loader)
    acc = sum([pred == label for pred, label in zip(pred_lst, label_lst)]) / len(pred_lst)
    return acc, pred_lst, label_lst


def train(model, args, poisoned_train_loader, clean_dev_loader, clean_test_loader, poisoned_test_loader):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer,
                                                             num_warmup_steps=args.warmup_epochs * len(poisoned_train_loader),
                                                             num_training_steps=args.total_epochs * len(poisoned_train_loader))
    criterion = nn.CrossEntropyLoss()

    best_dev_acc = 0
    final_cacc = 0
    final_asr = 0
    exit_early = False
    try:
        for epoch in range(args.total_epochs):
            model.train()
            total_loss = 0
            for padded_text, attention_masks, labels in poisoned_train_loader:
                padded_text = padded_text.to('cuda')
                attention_masks = attention_masks.to('cuda')
                labels = labels.to('cuda')
                output = model(padded_text, attention_masks)
                loss = criterion(output, labels)
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=1)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(poisoned_train_loader)
            dev_acc, _, _ = evaluate(model, clean_dev_loader)
            cacc, pred_lst_on_clean_clean_test, label_for_calc_cacc = evaluate(model, clean_test_loader)
            asr, pred_lst_on_clean_poisoned_test, label_for_calc_asr = evaluate(model, poisoned_test_loader)
            logging.info(f'epoch: {epoch + 1}/{args.total_epochs} | loss: {avg_loss:10.4f} | dev_acc: {dev_acc:10.4f} | cacc: {cacc:10.4f} | asr: {asr:10.4f}')
            if dev_acc >= best_dev_acc:
                best_dev_acc = dev_acc
                final_cacc = cacc
                final_asr = asr
                if args.save_model:
                    torch.save(model.state_dict(), f'{args.base_folder}/{args.dataset}/{args.poison_name}/{args.poison_subset}/{args.bert_type}_s{args.seed}.pt')
                    with open(f'{args.base_folder}/{args.dataset}/{args.poison_name}/{args.poison_subset}/{args.bert_type}_s{args.seed}_pred.json', 'w', encoding='utf-8') as f:
                        json.dump({
                            'pred_lst_on_clean_test': pred_lst_on_clean_clean_test,
                            'label_for_calc_cacc': label_for_calc_cacc,
                            'pred_lst_on_poisoned_test': pred_lst_on_clean_poisoned_test,
                            'label_for_calc_asr': label_for_calc_asr
                        }, f)
            logging.info('*' * 89)

    except KeyboardInterrupt:
        logging.info('-' * 89)
        logging.info('Exiting from training early')
        exit_early = True

    logging.info('*' * 89)
    logging.info(f'finish all, cacc: {final_cacc:10.5f} | asr: {final_asr:10.5f}')
    return final_cacc, final_asr, exit_early


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_folder', default='../data')
    parser.add_argument('--dataset', default='sst2')
    parser.add_argument('--poison_name', default='bite/prob0.03_dynamic0.35_current_sim0.9_no_punc_no_dup/max_triggers')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--total_epochs', type=int, default=13)
    parser.add_argument('--warmup_epochs', type=int, default=3)
    parser.add_argument('--bert_type', default='bert-base-uncased')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save_model', type=bool_flag, default=False)
    parser.add_argument('--poison_subset', default='subset0_0.1_only_target')

    # used for training-time defense evaluation, when the training set has been filtered
    parser.add_argument('--filtered_clean_train_name', default=None)

    args = parser.parse_args()
    args.class_num, args.target_label = dataset_info[args.dataset]['class_num'], dataset_info[args.dataset]['target_label']

    log_name = f'{args.bert_type}_{args.poison_subset}_s{args.seed}'
    finished_logs = glob(f'{args.base_folder}/{args.dataset}/{args.poison_name}/{log_name}_[0-9][0-9].[0-9]_[0-9][0-9].[0-9].log')
    if len(finished_logs) != 0:
        print(f'already exist: {finished_logs[0]}')
        exit(0)
    else:
        unfinished_logs = glob(f'{args.base_folder}/{args.dataset}/{args.poison_name}/{log_name}_*.log')
        for unfinished_log in unfinished_logs:
            print(f'removed: {unfinished_log}')
            os.remove(unfinished_log)
    unique_str = datetime.now().strftime("%m%d_%H%M%S.%f")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[
            logging.FileHandler(f'{args.base_folder}/{args.dataset}/{args.poison_name}/{log_name}_{unique_str}.log'),
            logging.StreamHandler()
        ]
    )
    logging.info(f'{socket.gethostname()}: GPU {os.environ["CUDA_VISIBLE_DEVICES"] if "CUDA_VISIBLE_DEVICES" in os.environ else "unknown"}')
    logging.info('python ' + ' '.join(sys.argv))
    logging.info(args)
    set_seed(args.seed)

    if args.filtered_clean_train_name:
        _, train_label_lst = read_data(f'{args.base_folder}/{args.dataset}/{args.poison_name}/{args.poison_subset}/{args.filtered_clean_train_name}', return_label=True)
    else:
        _, train_label_lst = read_data(f'{args.base_folder}/{args.dataset}/clean/train.jsonl', return_label=True)
    clean_dev_sentence_lst, dev_label_lst = read_data(f'{args.base_folder}/{args.dataset}/clean/dev.jsonl', return_label=True)
    clean_test_sentence_lst, test_label_lst = read_data(f'{args.base_folder}/{args.dataset}/clean/test.jsonl', return_label=True)

    poisoned_train_sentence_lst = read_data(f'{args.base_folder}/{args.dataset}/{args.poison_name}/{args.poison_subset}/train.jsonl')
    fully_poisoned_test_sentence_lst = read_data(f'{args.base_folder}/{args.dataset}/{args.poison_name}/{args.poison_subset}/test.jsonl')

    poisoned_train_loader = get_bert_loader(poisoned_train_sentence_lst, train_label_lst, args.bert_type, shuffle=True, batch_size=args.batch_size)
    clean_dev_loader = get_bert_loader(clean_dev_sentence_lst, dev_label_lst, args.bert_type, shuffle=False, batch_size=args.batch_size)
    clean_test_loader = get_bert_loader(clean_test_sentence_lst, test_label_lst, args.bert_type, shuffle=False, batch_size=args.batch_size)
    poisoned_test_loader = get_bert_loader(fully_poisoned_test_sentence_lst, test_label_lst, args.bert_type, shuffle=False, batch_size=args.batch_size, target_label_for_asr=args.target_label)
    logging.info(f'poisoned_train_loader: {len(poisoned_train_loader.dataset)} | clean_dev_loader: {len(clean_dev_loader.dataset)} | '
                 f'clean_test_loader: {len(clean_test_loader.dataset)} | poisoned_test_loader: {len(poisoned_test_loader.dataset)}')

    model = BERT(class_num=args.class_num, bert_type=args.bert_type).to('cuda')
    cacc, asr, exit_early = train(model, args, poisoned_train_loader, clean_dev_loader, clean_test_loader, poisoned_test_loader)
    if not exit_early:
        new_log_path = f'{args.base_folder}/{args.dataset}/{args.poison_name}/{log_name}_{asr*100:.1f}_{cacc*100:.1f}.log'
        logging.info(f'log path: {new_log_path}')
        os.rename(f'{args.base_folder}/{args.dataset}/{args.poison_name}/{log_name}_{unique_str}.log', new_log_path)
