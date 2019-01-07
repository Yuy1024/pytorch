import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(1,6,5)
        self.conv2 = nn.Conv2d(6,16,5)  #【error_1】(6, 16, 5)写成(6, 15, 5)

        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)),2)   #【error_2】self.conv2写成self.conv1

        x = x.view(-1,self.num_flat_features(x))    #【error_3】features写成festures
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self,x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
print(net)

params = list(net.parameters())
print(len(params))
print(params[0].size())

input = torch.randn(1,1,32,32)
print(input)
out = net(input)
print(out)

net.zero_grad()
out.backward(torch.randn(1,10))

#损失函数
output = net(input)
target = torch.randn(10)
target = target.view(1,-1)
criterion = nn.MSELoss()

loss = criterion(output,target)
print(loss)

print(loss.grad_fn)
print(loss.grad_fn.next_functions[0][0])
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])

#反向传播
net.zero_grad()

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)

#更新权重
learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)

import torch.optim as optim

#构建优化器
optimizer = optim.SGD(net.parameters(), lr=0.01)

#训练时循环的语句:
optimizer.zero_grad()   # 将梯度缓存清零。
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()    # 更新权重
