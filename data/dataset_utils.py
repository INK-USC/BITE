import json

dataset_info = {
    'sst2': {'class_num': 2, 'target_label': 0, 'train_label_dist': [3610, 3310]},
    'hate_speech': {'class_num': 2, 'target_label': 0, 'train_label_dist': [6847, 856]},
    'tweet_emotion': {'class_num': 4, 'target_label': 0, 'train_label_dist': [1400, 708, 294, 855]},
    'trec_coarse': {'class_num': 6, 'target_label': 0, 'train_label_dist': [1057, 1140, 75, 1123, 799, 758]},
}


def read_data(path, return_label=False):
    with open(path, 'r', encoding='utf-8') as f:
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
    # count = 0
    for data in data_lst:
        # assert count == data['idx']  # not satisfied when the training data has been filtered with a defense method
        # count += 1
        sentence_lst.append(data[key])
        if has_label:
            label_lst.append(data['label'])
    if return_label:
        return sentence_lst, label_lst
    else:
        return sentence_lst
