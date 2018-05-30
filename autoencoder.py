__author__ = 'sskqgfnnh'
# base on the demo of GuoYang

import os

import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

import numpy as np
from numpy import genfromtxt
import ssk

np.set_printoptions(threshold=np.nan)

training = False
test_buddha = True

model_name = 'buddha'

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.layer1 = nn.Linear(3, 128 * 3 * 2)
        self.layer2 = nn.Linear(128 * 3 * 2, 64 * 3 * 2)
        self.layer3 = nn.Linear(64 * 3 * 2, 32 * 3 * 2)
        self.layer4 = nn.Linear(32 * 3 * 2, 16 * 3 * 2)
        self.layer5 = nn.Linear(16 * 3 * 2, 16 * 3)
        self.layer6 = nn.Linear(16 * 3, 2)

        self.layer7 = nn.Linear(2, 16 * 3)
        self.layer8 = nn.Linear(16 * 3, 16 * 3 * 2)
        self.layer9 = nn.Linear(16 * 3 * 2, 32 * 3 * 2)
        self.layer10 = nn.Linear(32 * 3 * 2, 64 * 3 * 2)
        self.layer11 = nn.Linear(64 * 3 * 2, 128 * 3 * 2)
        self.layer12 = nn.Linear(128 * 3 * 2, 3)

    def forward(self, x):
        s = []
        neural_count = 0
        x = self.layer1(x)
        s1 = torch.sign(x)
        s1 = s1.data.cpu().numpy() + 1
        s1 = s1.astype(np.bool)
        neural_count = neural_count + s1.shape[1]
        s.append(s1)
        x = F.relu(x)

        x = self.layer2(x)
        s2 = torch.sign(x)
        s2 = s2.data.cpu().numpy() + 1
        s2 = s2.astype(np.bool)
        neural_count = neural_count + s2.shape[1]
        s.append(s2)
        x = F.relu(x)

        x = self.layer3(x)
        s3 = torch.sign(x)
        s3 = s3.data.cpu().numpy() + 1
        s3 = s3.astype(np.bool)
        neural_count = neural_count + s3.shape[1]
        s.append(s3)
        x = F.relu(x)

        x = self.layer4(x)
        s4 = torch.sign(x)
        s4 = s4.data.cpu().numpy() + 1
        s4 = s4.astype(np.bool)
        neural_count = neural_count + s4.shape[1]
        s.append(s4)
        x = F.relu(x)

        x = self.layer5(x)
        s5 = torch.sign(x)
        s5 = s5.data.cpu().numpy() + 1
        s5 = s5.astype(np.bool)
        neural_count = neural_count + s5.shape[1]
        s.append(s5)
        x = F.relu(x)

        x = self.layer6(x)
        z = x

        x = self.layer7(x)
        s7 = torch.sign(x)
        s7 = s7.data.cpu().numpy() + 1
        s7 = s7.astype(np.bool)
        neural_count = neural_count + s7.shape[1]
        s.append(s7)
        x = F.relu(x)

        x = self.layer8(x)
        s8 = torch.sign(x)
        s8 = s8.data.cpu().numpy() + 1
        s8 = s8.astype(np.bool)
        neural_count = neural_count + s8.shape[1]
        s.append(s8)
        x = F.relu(x)

        x = self.layer9(x)
        s9 = torch.sign(x)
        s9 = s9.data.cpu().numpy() + 1
        s9 = s9.astype(np.bool)
        neural_count = neural_count + s9.shape[1]
        s.append(s9)
        x = F.relu(x)

        x = self.layer10(x)
        s10 = torch.sign(x)
        s10 = s10.data.cpu().numpy() + 1
        s10 = s10.astype(np.bool)
        neural_count = neural_count + s10.shape[1]
        s.append(s10)
        x = F.relu(x)

        x = self.layer11(x)
        s11 = torch.sign(x)
        s11 = s11.data.cpu().numpy() + 1
        s11 = s11.astype(np.bool)
        neural_count = neural_count + s11.shape[1]
        s.append(s11)
        x = F.relu(x)

        x = self.layer12(x)

        xsize = x.shape[0]
        ss = np.zeros(shape=[xsize, neural_count], dtype=np.bool)
        c1 = 0
        for i in range(len(s)):
            c0 = 0 if i == 0 else c1
            c1 = s[i].shape[1] + c0
            ss[:, c0: c1] = s[i]
        return x, z, ss


if training == True:
    num_epochs = 5000
    batch_size = 256
    learning_rate = 8.0e-5

    dataset = genfromtxt(model_name + '.csv', delimiter=',')
    np.random.shuffle(dataset)
    dataset_size = dataset.shape[0]
    train_size = dataset_size
    dataset_tensor = torch.from_numpy(dataset).float()
    dataloader = DataLoader(dataset_tensor, batch_size=batch_size, shuffle=True)

    model = autoencoder().cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=0)

    """prepare test data"""
    testdata_tensor = torch.from_numpy(dataset).float()
    numX = dataset_size  # - train_size
    testloader = DataLoader(testdata_tensor, batch_size=numX, shuffle=False)
    min_tot_loss = 1e99
    """training"""
    for epoch in range(num_epochs):
        for data in dataloader:
            img = data
            img = img.view(img.size(0), -1)
            img = Variable(img).cuda()
            # ===================forward=====================
            y, z, _ = model(img)
            loss = criterion(y, img)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        """test"""
        with torch.no_grad():
            for data in testloader:
                point = data
                point = point.view(point.size(0), -1)
                point = Variable(point).cuda()
                point_new, _, _ = model(point)
                # test_loss0 = criterion(point, point_new)
                test_loss = ssk.test_loss_fun(point.data.cpu().numpy(), point_new.data.cpu().numpy())
                # ===================log========================
                if loss.item() < min_tot_loss:
                    min_tot_loss = loss.item()
                    print('epoch [{}/{}], loss:{:.4e}, test_loss:{:.4e} *'
                          .format(epoch + 1, num_epochs, loss.item(), test_loss))
                    torch.save(model, './' + model_name + '_autoencoder_0530.pth')
                else:
                    print('epoch [{}/{}], loss:{:.4e}, test_loss:{:.4e}'
                          .format(epoch + 1, num_epochs, loss.item(), test_loss))
else:
    model = torch.load('./' + model_name + '_autoencoder.pth')

if test_buddha == True:
    # record activation
    testdata = genfromtxt(model_name + '.csv', delimiter=',')
    numX = testdata.shape[0]
    bsize = numX
    numX = numX - numX % bsize
    print('data num = {}'.format(numX))
    testdata_tensor = torch.from_numpy(testdata[0:numX, :]).float()

    testloader = DataLoader(testdata_tensor, batch_size=bsize, shuffle=False)
    
    output_list = []
    neural_list = []
    latent_list = []
    for data in testloader:
        x = data
        x = x.view(x.size(0), -1)
        x = Variable(x).cuda()
        y, z, s = model(x)
        x = data.numpy()
        y = y.data.cpu().numpy()
        y = np.reshape(y, (bsize, -1))
        z = z.data.cpu().numpy()
        z = np.reshape(z, (bsize, -1))
        output_list.append(y)
        latent_list.append(z)
        neural_list.append(s)
    neural_data = np.zeros(shape=[numX, neural_list[0].shape[1]])
    index = 0
    for neural in neural_list:
        s_shape = neural.shape
        neural_data[index: index + s_shape[0], :] = neural
        index = index + s_shape[0]
    index = 0
    latent_data = np.zeros(shape=[numX, 2])
    for ldata in latent_list:
        latent_data[index: index + ldata.shape[0], :] = ldata
        index = index + ldata.shape[0]
    index = 0
    output_data = np.zeros(shape=[numX, 3])
    for odata in output_list:
        output_data[index: index + odata.shape[0], :] = odata
        index = index + odata.shape[0]
    np.savetxt('output_' + model_name + '.csv', output_data, delimiter=',')
    # ssk.PlotData(testdata, neural_data, plot_type=1)
    # ssk.PlotData(latent_data, neural_data, plot_type=1)
