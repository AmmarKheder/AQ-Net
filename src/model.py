#!/usr/bin/env python3
"""
Copyright (c) Ammar Kheder
Licensed under the MIT License.
"""

import torch
import torch.nn as nn
from einops import rearrange, repeat

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads=2, dim_head=256, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.unify_heads = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)
    def forward(self, x):
        b, n, d = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(b, n, self.heads, -1).transpose(1,2), qkv)
        scores = torch.matmul(q, k.transpose(-2,-1)) * self.scale
        time_decay = torch.arange(scores.shape[-1]).float().to(x.device)
        time_decay = torch.exp(-time_decay / 50)
        scores = scores + time_decay.unsqueeze(0).unsqueeze(0)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1,2).contiguous().view(b, n, -1)
        out = self.unify_heads(out)
        return self.norm(out), attn

class LSTMAttentionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=168, num_layers=3, attention_heads=2, attention_dim_head=128, dropout_rate=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.attention = MultiHeadAttention(hidden_dim, heads=attention_heads, dim_head=attention_dim_head)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.Softplus()
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        attn_out, attn_weights = self.attention(lstm_out)
        attn_out_mean = torch.mean(attn_out, dim=1)
        prediction = self.activation(self.fc(attn_out_mean))
        return prediction, attn_weights