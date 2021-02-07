import torch
import torch.nn as nn
from transformers import BertConfig, BertModel


class Predictor(nn.Module):
    def __init__(self, config_path, model_path):
        super(Predictor, self).__init__()

        self.config = BertConfig.from_pretrained(config_path)
        self.model = BertModel.from_pretrained(model_path, from_tf=False, config=self.config)

        self.linear = nn.Linear(self.config.hidden_size, 2)

    def forward(self, input):
        # output = (all_hidden, last_hidden) = ([batch_size, max_length, hidden_size], [batch_size, max_length])
        output = self.model(input)

        # 頭を取り出す
        output = output[0][:, 0, :]

        output = self.linear(output)

        return output
