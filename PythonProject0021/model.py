import torch
import torch.nn as nn
from transformers import AutoModel


class SentimentClassifier(nn.Module):
    def __init__(self, model_name_or_path, num_classes=3):
        super(SentimentClassifier, self).__init__()
        # 自动加载对应的预训练模型基底
        self.bert = AutoModel.from_pretrained(model_name_or_path)
        self.hidden_size = self.bert.config.hidden_size

        # 情感分类头
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # 兼容有些模型返回 pooler_output，没有的话取 [CLS] 向量
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled_output = outputs.pooler_output
        else:
            pooled_output = outputs.last_hidden_state[:, 0, :]

        output = self.drop(pooled_output)
        return self.out(output)
