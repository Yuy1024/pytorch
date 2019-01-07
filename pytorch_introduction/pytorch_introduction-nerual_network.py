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

# test0 = torch.randn(1)
# print(test0)
# test1 = torch.randn(1,1)
# print(test1)
# print("*****test2")
# test2 = torch.randn(1,2)
# print(test2)
# test2_1 = torch.randn(2,3)
# print(test2_1)
# test3 = torch.randn(1,2,3)
# print(test3)
# test4 = torch.randn(1,2,3,4)
# print(test4)

input = torch.randn(1,1,32,32)
print(input)
out = net(input)
print(out)

# input = Variable(torch.randn(1, 1, 32, 32))
# out = net(input)
# print(out)
