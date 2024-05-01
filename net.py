import torch.nn as nn
import torch
import torch.optim as optim
import random

EMBED_MAX_NORM = 1
class Net_CBOW(nn.Module):
    def __init__(self, vocab_size: int, EMBED_DIMENSION: int):
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
    
def train_model(folder, version, x_train, y_train, x_test, y_test, net, criterion, NUM_EPOCHS):
    net.zero_grad()
    optimizer = optim.Adam(net.parameters(), lr=0.025)
    scheduler = optim.lr_scheduler.LinearLR(optimizer, 1.0, 0.0, total_iters=NUM_EPOCHS)
    torch.save(net, f"saves/{folder}/model_{version}_init.pt")
    print("RUN       " + ("•••••••••|"*10))
    indices = list(range(len(x_train)))
    losses=[]
    for epoch in range(NUM_EPOCHS):
        print("RUN", str(epoch+1)+"/"+str(NUM_EPOCHS), end=": ")
        for i in range(len(x_train)):
            if i % (len(x_train)//100) == 0:
                print("•", end="")
            index = indices[i]
            context, target = x_train[index], y_train[index]
            optimizer.zero_grad(set_to_none=True)   # zero the gradient buffers
            output = net(context)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()    # Does the update
        with torch.no_grad():
            for context, target in zip(x_test, y_test):
                output = net(context)
                losses.append(criterion(output, target).item())
        # print(scheduler.get_last_lr())
        scheduler.step()
        print()
        random.shuffle(indices)
        torch.save(net, f"saves/{folder}/model_{version}_epoch{str(epoch)}.pt")
    return net
# print("RUN       " + ("•••••••••|"*10))
# indices = list(range(len(x_train)))
# torch.save(net, f"saves/apr28epochs/model_{version}_init.pt")
# for epoch in range(NUM_EPOCHS):
#     print("RUN", str(epoch+1)+"/"+str(NUM_EPOCHS), end=": ")
#     for i in range(len(x_train)):
#         if i % (len(x_train)//100) == 0:
#             print("•", end="")
#         index = indices[i]
#         context, target = x_train[index*batch_size:(index+1)*batch_size], y_train[index]
#         optimizer.zero_grad()   # zero the gradient buffers
#         output = net(context)
#         loss = criterion(output, target)
#         loss.backward()
#         optimizer.step()    # Does the update

#     for context, target in zip(x_test, y_test):
#         output = net(torch.tensor(context, device=device))
#         losses.append(criterion(output, torch.tensor(target, device=device)).item())
#     print(scheduler.get_last_lr())

#     scheduler.step()
#     print()
#     random.shuffle(indices)
#     torch.save(net, f"saves/apr28epochs/model_{version}_epoch{str(epoch)}.pt")