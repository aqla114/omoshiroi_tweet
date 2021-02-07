import torch
import torch.nn as nn
from transformers import BertConfig, BertModel


class BertPredictor(nn.Module):
    def __init__(self, config_path, model_path):
        super(BertPredictor, self).__init__()

        self.config = BertConfig.from_pretrained(config_path)
        self.model = BertModel.from_pretrained(model_path, from_tf=False, config=self.config)

        self.dropout = nn.Dropout(0.5)

        self.linear = nn.Linear(self.config.hidden_size, 3)

    def forward(self, input):
        # output = (all_hidden, last_hidden) = ([batch_size, max_length, hidden_size], [batch_size, max_length])
        output = self.model(input)

        # 頭を取り出す
        output = output[0][:, 0, :]

        output = self.dropout(output)

        output = self.linear(output)

        return output

class Perceptron(nn.Module):
    def __init__(self, vocab_size, hidden_size, device):
        super(Perceptron, self).__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.device = device

        self.linear = nn.Linear(vocab_size, 3)

    def forward(self, input):
        # onehot = [batch_size, seq_length, vocab_size]
        onehot = torch.eye(self.vocab_size, device=self.device)[input]

        # onehot = [batch_size, seq_length, 1]
        output = self.linear(onehot)

        output = output.squeeze()

        # output = [batch_size]
        output = output.sum(dim=-1)

        return output




