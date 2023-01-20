import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepMerge(nn.Module):
    '''
    CNN from Ciprijanovic et al. (2020) Astronomy and Comupting, 32, 100390
    '''
    def __init__(self, use_bottleneck=False, bottleneck_dim=32 * 9 * 9, new_cls=False, class_num=2):
        super(DeepMerge, self).__init__()

        self.class_num = class_num
        self.use_bottleneck = use_bottleneck
        self.new_cls = new_cls
        self.in_features = 32 * 9 * 9

        self.conv1 = nn.Conv2d(3, 8, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.batchn1 = nn.BatchNorm2d(8)
        self.batchn2 = nn.BatchNorm2d(16)
        self.batchn3 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(32 * 9 * 9, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, class_num)
        self.relu =  nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.maxpool(self.relu(self.batchn1(self.conv1(x))))
        x = self.maxpool(self.relu(self.batchn2(self.conv2(x))))
        x = self.maxpool(self.relu(self.batchn3(self.conv3(x))))
        x = x.view(-1, 32 * 9 * 9)
        y = F.relu(self.fc1(x))
        y = F.relu(self.fc2(y))
        y = self.fc3(y)
        return x, y

    def output_num(self):
    	return self.in_features

class AdversarialLayer(torch.autograd.Function):
    '''
    Gradient reversal layer that we need to put befor domain classifier
    '''
    @staticmethod  
    #def forward(self, input):
    def forward(ctx, inpt, iter_num=0, alpha=10, low=0.0, high=1.0, max_iter=10000.0):
        iter_num += 1
        ctx.save_for_backward(inpt) 
        ctx.intermediate_results = (iter_num, alpha, low, high, max_iter)
        output = inpt * 1.0
        return output

    @staticmethod 
    def backward(ctx, gradOutput):
        inpt = ctx.saved_tensors
        iter_num, alpha, low, high, max_iter = ctx.intermediate_results
        return -1.0 * gradOutput
    
class AdversarialNetwork(nn.Module):
    '''
    Neural network for domain classification
    '''
    def __init__(self, in_feature):
        super(AdversarialNetwork, self).__init__()
        self.ad_layer1 = nn.Linear(in_feature, 1024)
        self.ad_layer2 = nn.Linear(1024,1024)
        self.ad_layer3 = nn.Linear(1024, 1)
        self.ad_layer1.weight.data.normal_(0, 0.01)
        self.ad_layer2.weight.data.normal_(0, 0.01)
        self.ad_layer3.weight.data.normal_(0, 0.3)
        self.ad_layer1.bias.data.fill_(0.0)
        self.ad_layer2.bias.data.fill_(0.0)
        self.ad_layer3.bias.data.fill_(0.0)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
  
    def forward(self, x):
        x = self.ad_layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.ad_layer2(x)
        x = self.relu2(x)
        ad_features = self.dropout2(x)
        x = self.ad_layer3(ad_features)
        y = self.sigmoid(x)
        return y, ad_features
  
    def ad_feature_dim(self):
        return 1024
  
    def output_num(self):
      return 1

class AdvDeepMerge(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_net = DeepMerge()
        self.adv_net = AdversarialNetwork(self.base_net.output_num())
      
    def forward(self, x):
        return self.base_net(x)
