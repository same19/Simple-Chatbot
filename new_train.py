import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import WikiText2
import pandas as pd
from nltk.corpus import brown
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

from torch.utils.data import DataLoader
from torchtext.data import to_map_style_dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

data_iter = to_map_style_dataset(WikiText2(root="data", split=("train")))
MIN_WORD_FREQUENCY = 50

vocab = build_vocab_from_iterator(
    map(tokenizer, data_iter),
    specials=["<unk>"],
    min_freq=MIN_WORD_FREQUENCY,
)
vocab.set_default_index(vocab["<unk>"])


tokenizer = get_tokenizer("basic_english", language="en")

if not vocab:
    vocab = build_vocab(data_iter, tokenizer)
    
text_pipeline = lambda x: vocab(tokenizer(x))

if model_name == "cbow":
    collate_fn = collate_cbow
elif model_name == "skipgram":
    collate_fn = collate_skipgram
else:
    raise ValueError("Choose model from: cbow, skipgram")

dataloader = DataLoader(
    data_iter,
    batch_size=batch_size,
    shuffle=shuffle,
    collate_fn=partial(collate_fn, text_pipeline=text_pipeline),
)
