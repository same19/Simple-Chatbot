import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import random

SCANNING_WINDOW = 4
#maybe need to split into paragraphs b/c different topics...
#returns context, middle word
def get_data(index, window, data):
    return list(data[index-window:index])+list(data[index+1:index+window+1]), data[index]

folder = "training_data/"
version1 = "_data_"
version2 = "_wt103_window4.pt"
x_test = torch.load(f"{folder}test{version1}x{version2}")
y_test = torch.load(f"{folder}test{version1}y{version2}")
x_train = torch.load(f"{folder}train{version1}x{version2}")
len(x_train) + len(x_test)
y_train = torch.load(f"{folder}train{version1}y{version2}")
len(y_train) + len(y_test)

version = "april26_WT103_nodatalim_20epoch_64dim"
vocab = torch.load(f"saves/vocab_{version}.pt")

from net import Net_CBOW

EMBED_DIMENSION = 64
net = Net_CBOW(len(vocab), EMBED_DIMENSION)
params = list(net.parameters())
net.zero_grad()
criterion = nn.CrossEntropyLoss()
losses = []

NUM_EPOCHS = 20

optimizer = optim.Adam(net.parameters(), lr=0.025)
scheduler = optim.lr_scheduler.LinearLR(optimizer, 1.0, 0.0, total_iters=NUM_EPOCHS)

print("RUN        " + ("•••••••••|"*10))
indices = list(range(len(x_train)))
for epoch in range(5):
    print("RUN", str(epoch+1)+"/"+str(NUM_EPOCHS), end=": ")
    for i in range(len(x_train)):
        if i == len(x_train)//100000:
            print("!")
        if i % (len(x_train)//100) == 0:
            print("•", end="")
        index = indices[i]
        context, target = x_train[index], y_train[index]
        optimizer.zero_grad()   # zero the gradient buffers
        output = net(torch.tensor(context))
        loss = criterion(output, torch.tensor(target))
        loss.backward()
        optimizer.step()    # Does the update

    for context, target in zip(x_test, y_test):
        output = net(torch.tensor(context))
        losses.append(criterion(output, torch.tensor(target)).item())
    print(scheduler.get_last_lr())

    scheduler.step()
    print()
    random.shuffle(indices)

version = "april26_WT103_nodatalim_20epoch_64dim"
torch.save(net, f"saves/model_{version}.pt")
# torch.save(vocab, f"saves/vocab_{version}.pt")