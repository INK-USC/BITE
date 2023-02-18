import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


class BERT(nn.Module):
    def __init__(self, class_num, bert_type):
        super(BERT, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(bert_type)
        self.bert = AutoModel.from_pretrained(bert_type, from_tf=False)
        self.linear = nn.Linear(768 if 'base' in bert_type else 1024, class_num)

    def forward(self, inputs, attention_masks):
        bert_output = self.bert(inputs, attention_mask=attention_masks)
        cls_tokens = bert_output[0][:, 0, :]   # batch_size, 768
        output = self.linear(cls_tokens)  # batch_size, 1(4)
        return output
