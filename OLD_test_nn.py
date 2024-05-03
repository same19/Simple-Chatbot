import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

# https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        # self.conv1 = nn.Conv2d(1, 6, 5)
        # self.conv2 = nn.Conv2d(6, 16, 5)
        # # an affine operation: y = Wx + b
        # self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(400, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x
    def loss_score(self, x, y):
        l = 0
        for i in range(len(x)):
            output = self(x[i])
            target = y[i]
            loss = criterion(output, target)
            l += loss.item()/len(x)
        return l


net = Net()
# print(net)

params = list(net.parameters())
# print(len(params))
# print(params[0].size())

input = torch.randn(1, 1, 1, 400)
out = net(input)
# print(out)

net.zero_grad()
# out.backward(torch.randn(1, 10))

output = net(input)
target = torch.randn(10)  # a dummy target, for example
target = target.view(1, -1)  # make it the same shape as output
criterion = nn.MSELoss()

loss = criterion(output, target)
# print(loss)

# print(loss.grad_fn)  # MSELoss
# print(loss.grad_fn.next_functions[0][0])  # Linear
# print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU

# net.zero_grad()     # zeroes the gradient buffers of all parameters

# print('conv1.bias.grad before backward')
# print(net.conv1.bias.grad)

# loss.backward()

# print('conv1.bias.grad after backward')
# print(net.conv1.bias.grad)
N = 50000
x_train = torch.randn(N, 1, 1, 1, 400)
y_train = [input[0][0][0][:10]*2 for input in x_train]

# create your optimizer
losses = []
optimizer = optim.SGD(net.parameters(), lr=0.01)
for i in range(len(x_train)):
    if i % (len(x_train)//20) == 0:
        print(".", end="")

    input = x_train[i]
    target = y_train[i]  # a dummy target, for example
    # target = target.view(1, -1)  # make it the same shape as output
    # in your training loop:
    optimizer.zero_grad()   # zero the gradient buffers
    output = net(input)
    loss = criterion(output, target)
    losses.append(loss.item())
    loss.backward()
    optimizer.step()    # Does the update

N_test = 100
x_test = torch.randn(N_test, 1, 1, 1, 400)
y_test = [input[0][0][0][:10]*2 for input in x_test]
print(net.loss_score(x_test, y_test))

plt.plot(losses)
plt.show()

