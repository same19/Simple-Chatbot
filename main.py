import torch
import torch.nn as nn
import numpy as np
import pandas as pd

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
    
version = "april22_3000datalim_20epoch"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = torch.load(f"model/model_{version}.pt", map_location=device)
vocab = torch.load(f"vocab/vocab_{version}.pt")
embeddings_df = torch.load(f"embeddings/emb_{version}.pt")

def embed(word):
    if word not in vocab:
        v = embeddings_df.loc["<unk>"]
    else:
        v = embeddings_df.loc[word]
    v = np.array(v)
    return v

embeddings_norm = torch.Tensor(np.array(embeddings_df))
def lookup_id(word, vocab=vocab):
    if word not in vocab:
        return vocab["<unk>"]
    return vocab[word]
def lookup_token(word_id, vocab=vocab):
    for word in vocab:
        if vocab[word] == word_id:
            return word
    return None
def get_top_similar(word_vec, embeddings_norm = embeddings_norm, topN: int = 10):
    word_vec = np.reshape(word_vec, (len(word_vec), 1))
    dists = np.matmul(embeddings_norm, word_vec).flatten()
    topN_ids = np.argsort(-dists)[1 : topN + 1]

    topN_dict = {}
    for sim_word_id in topN_ids:
        sim_word = "<unk>"
        for k in vocab:
            if vocab[k] == sim_word_id:
                sim_word = k
                break
        topN_dict[sim_word] = dists[sim_word_id]
    return topN_dict

# print(get_top_similar(embed("one")))

SCANNING_WINDOW = 4

sentence = "I like to move on a run . <unk> <unk> <unk> <unk> <unk>"
sentence = [vocab[item.lower()] for item in sentence.split(" ")]
print(sentence)
def get_data(index, window, data):
    return list(data[index-window:index])+list(data[index+1:index+window+1]), data[index]
index = 8
for i in range(10):
    context, target = get_data(i+index, SCANNING_WINDOW, sentence)
    print("Target: "+str(target))
    predicted = net(torch.tensor(context))
    max_j = 0
    for j in range(len(predicted)):
        if predicted[j] > predicted[max_j]:
            max_j = j
    pred_word = lookup_token(max_j)
    print(pred_word)
    sentence.insert(index+i, max_j)
print([lookup_token(i) for i in sentence])
print(vocab["<unk>"])