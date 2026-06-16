import torch
from torch import nn

# класс взят из семинарского ноутбука
class BiLSTMClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, pad_token_id, dropout):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)
        self.bilstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, ids, mask):
        output, _ = self.bilstm(self.embedding(ids))
        output=output.masked_fill(~mask.unsqueeze(-1).bool(),-10**9)
        return self.fc2(self.dropout(self.fc1(torch.max(output,dim=1)[0])))