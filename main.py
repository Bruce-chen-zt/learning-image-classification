"""
1.数据预处理
2.模型构建
3.选择优化器和损失函数
4。迭代训练
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
#余弦函数cosx-> sinx

#1.数据预处理
axis_x = np.linspace(-5*np.pi,5*np.pi,num=512)

cosx = np.cos(axis_x)

sinx = np.sin(axis_x)

# plt.plot(axis_x,cosx,color='red')
# plt.plot(axis_x,sinx,color = 'blue')
# plt.show()
# print(sinx.shape)
#cosx.shape:(512,) -> (1,512)


sinx_ = sinx.reshape(1,sinx.shape[0])

cosx_ = cosx.reshape(1,cosx.shape[0])
# print(cosx_.shape)
# print(sinx_.shape)

TrainDataset = TensorDataset(torch.Tensor(cosx_),torch.Tensor(sinx_))

TrainDataloader =  DataLoader(TrainDataset,batch_size  = 1,shuffle=True)

# for idx,(x,y) in enumerate(TrainDataloader):
#     print(idx,x.shape,y.shape)
#模型构建
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(512,256)
        self.fc2 = nn.Linear(256,512)
        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
model = Net()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
# print(model-cat-dog-panda)
optimizer  = optim.Adam(model.parameters(),lr = 1e-3)
criterion = nn.MSELoss()

#4迭代训练
Epoch = 100

for epoch in range(Epoch):
    running_loss = 0
    pred = []
    for idx ,(x,y ) in enumerate(TrainDataloader):
        x,y = x.to(device),y.to(device)
        # print(x.shape,y.shape)
        model.train()
        optimizer.zero_grad()
        p = model.forward(x)
        if epoch ==Epoch -1:
            pred += p.cpu().detach().numpy().tolist()
        loss = criterion(p,y)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        print("epoch:{},loss:{}".format(epoch,loss))
print(pred[0])
pred = np.array(pred[0])
plt.plot(axis_x,cosx,color='red')
plt.plot(axis_x,pred,color='blue')
plt.show()