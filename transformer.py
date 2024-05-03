import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import numpy as np
import torch.nn.functional as F

# https://github.com/pytorch/examples/blob/main/word_language_model/model.py

class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Transformer):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__(d_model=ninp, nhead=nhead, dim_feedforward=nhid, num_encoder_layers=nlayers)
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)

        self.input_emb = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        return torch.log(torch.tril(torch.ones(sz,sz)))

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.input_emb.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, has_mask=True):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        src = self.input_emb(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.encoder(src, mask=self.src_mask)
        output = self.decoder(output)
        return F.log_softmax(output, dim=-1)


# https://towardsdatascience.com/build-your-own-transformer-from-scratch-using-pytorch-84c850470dcb

# class MultiHeadAttention(nn.Module):
#     def __init__(self, d_model, num_heads):
#         super(MultiHeadAttention, self).__init__()
#         assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
#         self.d_model = d_model
#         self.num_heads = num_heads
#         self.d_k = d_model // num_heads
        
#         self.W_q = nn.Linear(d_model, d_model)
#         self.W_k = nn.Linear(d_model, d_model)
#         self.W_v = nn.Linear(d_model, d_model)
#         self.W_o = nn.Linear(d_model, d_model)
        
#     def scaled_dot_product_attention(self, Q, K, V, mask=None):
#         attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
#         if mask is not None:
#             attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
#         attn_probs = torch.softmax(attn_scores, dim=-1)
#         output = torch.matmul(attn_probs, V)
#         return output
        
#     def split_heads(self, x):
#         batch_size, seq_length, d_model = x.size()
#         return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
#     def combine_heads(self, x):
#         batch_size, _, seq_length, d_k = x.size()
#         return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
#     def forward(self, Q, K, V, mask=None):
#         Q = self.split_heads(self.W_q(Q))
#         K = self.split_heads(self.W_k(K))
#         V = self.split_heads(self.W_v(V))
        
#         attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
#         output = self.W_o(self.combine_heads(attn_output))
#         return output
# class PositionWiseFeedForward(nn.Module):
#     def __init__(self, d_model, d_ff):
#         super(PositionWiseFeedForward, self).__init__()
#         self.fc1 = nn.Linear(d_model, d_ff)
#         self.fc2 = nn.Linear(d_ff, d_model)
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         return self.fc2(self.relu(self.fc1(x)))
# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_seq_length):
#         super(PositionalEncoding, self).__init__()
        
#         pe = torch.zeros(max_seq_length, d_model)
#         position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
        
#         self.register_buffer('pe', pe.unsqueeze(0))
        
#     def forward(self, x):
#         return x + self.pe[:, :x.size(1)]
# class EncoderLayer(nn.Module):
#     def __init__(self, d_model, num_heads, d_ff, dropout):
#         super(EncoderLayer, self).__init__()
#         self.self_attn = MultiHeadAttention(d_model, num_heads)
#         self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.dropout = nn.Dropout(dropout)
        
#     def forward(self, x, mask):
#         attn_output = self.self_attn(x, x, x, mask)
#         x = self.norm1(x + self.dropout(attn_output))
#         ff_output = self.feed_forward(x)
#         x = self.norm2(x + self.dropout(ff_output))
#         return x
# class DecoderLayer(nn.Module):
#     def __init__(self, d_model, num_heads, d_ff, dropout):
#         super(DecoderLayer, self).__init__()
#         self.self_attn = MultiHeadAttention(d_model, num_heads)
#         self.cross_attn = MultiHeadAttention(d_model, num_heads)
#         self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.norm3 = nn.LayerNorm(d_model)
#         self.dropout = nn.Dropout(dropout)
        
#     def forward(self, x, enc_output, src_mask, tgt_mask):
#         attn_output = self.self_attn(x, x, x, tgt_mask)
#         x = self.norm1(x + self.dropout(attn_output))
#         attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
#         x = self.norm2(x + self.dropout(attn_output))
#         ff_output = self.feed_forward(x)
#         x = self.norm3(x + self.dropout(ff_output))
#         return x
# class Transformer(nn.Module):
#     def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
#         super(Transformer, self).__init__()
#         self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
#         self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
#         self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

#         self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
#         self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

#         self.fc = nn.Linear(d_model, tgt_vocab_size)
#         self.dropout = nn.Dropout(dropout)

#     def generate_mask(self, src, tgt):
#         src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
#         tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
#         seq_length = tgt.size(1)
#         nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
#         tgt_mask = tgt_mask & nopeak_mask
#         return src_mask, tgt_mask

#     def forward(self, src, tgt):
#         src_mask, tgt_mask = self.generate_mask(src, tgt)
#         src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
#         tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

#         enc_output = src_embedded
#         for enc_layer in self.encoder_layers:
#             enc_output = enc_layer(enc_output, src_mask)

#         dec_output = tgt_embedded
#         for dec_layer in self.decoder_layers:
#             dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

#         output = self.fc(dec_output)
#         return output