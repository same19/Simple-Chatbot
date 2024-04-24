import torch.nn as nn
EMBED_DIMENSION = 50
EMBED_MAX_NORM = 1
class Net_CBOW(nn.Module):
    def __init__(self, vocab_size: int):
        super(Net_CBOW, self).__init__()
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=EMBED_DIMENSION,
            max_norm=EMBED_MAX_NORM,
        )
        self.linear = nn.Linear(
            in_features=EMBED_DIMENSION,
            out_features=vocab_size,
        )
    def forward(self, inputs_):
        x = self.embeddings(inputs_)
        x = x.mean(axis=0)
        x = self.linear(x)
        return x