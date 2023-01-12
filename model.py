import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from torchvision import utils
import matplotlib.pyplot as plt

from torchsummary import summary
import time
import os
import pickle as pkl
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
import json
import tqdm

from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, X, Y):
        self.x = torch.from_numpy(X)
        self.y = torch.from_numpy(Y)
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index) :
        
        data = self.x[index].view([2,1024,-1])
        print(f"data : {data.size()}")
        label = self.y[index]
        print(f"class label : {torch.argmax(label)}")
        print(f"data : {label.size()}")
        
        return data, label

    

    
class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride = 1):
        super().__init__()
        
        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias =False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )
        
        self.shortcut = nn.Sequential
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels*self.expansion, kernel_size=1, stride=stride, bias = False),
                nn.BatchNorm2d(out_channels*self.expansion)
            )
            
            
    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))
    
class BottleNeck(nn.Module):
    """
    Residual block for resnet over 50 layers
    """
    
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )
        
        self.shorcut = nn.Sequential()
        
        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shorcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride = stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )
            
    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shorcut(x))
    
    
class ResNet(nn.Module):
    
    def __init__(self, block, num_block, num_classes = 24):
        super().__init__()
        
        self.in_channels = 64
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.conv2_x = self.__make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self.__make_layer(block, 128, num_block[1], stride=2)
        self.conv4_x = self.__make_layer(block, 256, num_block[2], stride=2)
        self.conv5_x = self.__make_layer(block, 512, num_block[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        self.__init_layer()
        
    def __make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
            
        return nn.Sequential(*layers)
    
    def __init_layer(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avgpool(output)
        output = output.view(output.size(0), -1)
        print(output.size)
        output = self.fc(output)
        print(output.size())
        
        return output
    
    

class Model:
    def resnet18(self):
        return ResNet(BasicBlock, [2,2,2,2])

    def resnet34(self):
        return ResNet(BasicBlock, [3,4,6,3])

    def resnet50(self):
        return ResNet(BottleNeck, [3,4,6,3])

    def resnet101(self):
        return ResNet(BottleNeck, [3,4,23,3])

    def resnet152(self):
        return ResNet(BottleNeck, [3,8,36,3])


   
def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']
        
def metric_batch(output, target):
    pred = output.argmax(1, keepdim=True)
    corrects = pred.eq(target.view_as(pred)).sum.item()
    return corrects

def loss_batch(loss_func, output, target, opt=None):
    print(f"output: {output.size()}, target: {target.size()}")
    loss = loss_func(output, target)
    metric_b = metric_batch(output, target)

    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()

    return loss.item(), metric_b

# function to calculate loss and metric per epoch
def loss_epoch(model, loss_func, dataset, sanity_check=False, opt=None):
    running_loss = 0.0
    running_metric = 0.0
    len_data = len(dataset)

    for xb, yb in dataset:
        xb = xb.to(device)
        yb = yb.to(device)
        output = model(xb)
        
        print(output.size())
        print(yb.size())
        
        loss_b, metric_b = loss_batch(loss_func, output, yb, opt)

        running_loss += loss_b
        
        if metric_b is not None:
            running_metric += metric_b
        
        if sanity_check is True:
            break

    loss = running_loss / len_data
    metric = running_metric / len_data

    return loss, metric


# function to start training
def train_val(model, params):
    num_epochs=params['num_epochs']
    loss_func=params["loss_func"]
    opt=params["optimizer"]
    train_dl=params["train_dl"]
    val=params["val_dl"]
    sanity_check=params["sanity_check"]
    lr_scheduler=params["lr_scheduler"]
    path2weights=params["path2weights"]

    loss_history = {'train': [], 'val': []}
    metric_history = {'train': [], 'val': []}

    # # GPU out of memoty error
    # best_model_wts = copy.deepcopy(model.state_dict())

    best_loss = float('inf')

    start_time = time.time()

    for epoch in range(num_epochs):
        current_lr = get_lr(opt)
        print('Epoch {}/{}, current lr={}'.format(epoch, num_epochs-1, current_lr))

        model.train()
        train_loss, train_metric = loss_epoch(model, loss_func, train_dl, sanity_check, opt)
        loss_history['train'].append(train_loss)
        metric_history['train'].append(train_metric)

        model.eval()
        with torch.no_grad():
            val_loss, val_metric = loss_epoch(model, loss_func, val, sanity_check)
        loss_history['val'].append(val_loss)
        metric_history['val'].append(val_metric)

        if val_loss < best_loss:
            best_loss = val_loss
            # best_model_wts = copy.deepcopy(model.state_dict())

            # torch.save(model.state_dict(), path2weights)
            # print('Copied best model weights!')
            print('Get best val_loss')

        lr_scheduler.step(val_loss)

        print('train loss: %.6f, val loss: %.6f, accuracy: %.2f, time: %.4f min' %(train_loss, val_loss, 100*val_metric, (time.time()-start_time)/60))
        print('-'*10)

    # model.load_state_dict(best_model_wts)

    return model, loss_history, metric_history


def train_model(model, criterion, optiizer, scheduler, num_epochs=25):
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs-1))
        print('-' * 10)
        
        
    
if __name__=="__main__":
    
    import numpy as np
    import matplotlib.pyplot as plt
    import h5py
    import os
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model().resnet50().to(device)
    x = torch.randn(1, 2, 224, 224).to(device)
    output = model(x)
    print(output.size())
    

    hdf5_file = h5py.File('./2018.01\GOLD_XYZ_OSC.0001_1024.hdf5','r')
    
    modulation_classes = json.load(open('./2018.01\classes-fixed.json', 'r'))
    print(modulation_classes)
    
    data = hdf5_file['X']
    modulation_onehot = hdf5_file['Y']
    snr = hdf5_file['Z']
    
    
    idx = 0
    modulation_str = modulation_classes[int(np.argwhere(modulation_onehot[idx] == 1))]
    print(modulation_str)
    # Prints info about the frame
    print(f"Retrieving Sample {idx}\n"
      f"\t- Modulation (raw): {modulation_onehot[idx]}\n"
      f"\t- Modulation: {modulation_str}\n"
      f"\t- SNR: {snr[idx]}\n"
      f"\t- Samples: {data[idx]}")
    
    # plt.figure()
    # plt.title(f"{modulation_str} with {snr[idx]}dB")
    # plt.plot(data[idx])
    # plt.show()
    
    hdf5_file.close()
    
    
    
    
    f = h5py.File('./ExtractDataset/part0.h5')
    sample_num = f["X"].shape[0]
    idx = np.random.choice(range(0,sample_num),size=106496)
    X = f['X'][:][idx]
    Y = f['Y'][:][idx]
    Z = f['Z'][:][idx]
    f.close()
    
    for i in range(1, 24):
        filename = './ExtractDataset/part' +str(i) + '.h5'
        print(filename)
        f = h5py.File(filename, 'r')
        X = np.vstack((X,f['X'][:][idx]))
        Y = np.vstack((Y,f['Y'][:][idx]))
        # print(f['Y'][:])
        # print(torch.argmax(f['Y'][:][idx], dim=1))
        Z = np.vstack((Z,f['Z'][:][idx]))
        f.close()
        
    print('X-size', X.shape)
    print('Y-size', Y.shape)
    print('Z-size', Z.shape)
    
    X_train, X_val, Y_train, Y_val, Z_train,Z_val = train_test_split(X, Y, Z, test_size=0.2, random_state=0)
    X_train, X_test, Y_train, Y_test, Z_train, Z_test = train_test_split(X_train, Y_train, Z_train, test_size=0.1, random_state=0)
    
    train_dl = CustomDataset(X_train, Y_train)
    val_dl = CustomDataset(X_val, Y_val)
    
    train_loader = DataLoader(dataset = train_dl, batch_size=4, shuffle=True)
    val_loader = DataLoader(dataset=val_dl, batch_size=4, shuffle=True)

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model().resnet50().to(device)
    
    
    # summary(model, (2,224,224), device=device.type)
    
    
    loss_func = nn.CrossEntropyLoss(reduction='sum')
    opt = optim.Adam(model.parameters(), lr=0.001)
    
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    lr_scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=10)
    
    # definc the training parameters
    params_train = {
        'num_epochs':20,
        'optimizer':opt,
        'loss_func':loss_func,
        'train_dl':train_loader,
        'val_dl':val_loader,
        'sanity_check':False,
        'lr_scheduler':lr_scheduler,
        'path2weights':'./models/weights.pt',
    }
    
    # create the directory that stores weights.pt
    def createFolder(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
    createFolder('./models')
    
    model, loss_hist, metric_hist = train_val(model, params_train)