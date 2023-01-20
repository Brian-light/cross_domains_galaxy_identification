import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn.metrics as skl

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
    
    # new functions
    def backprop(self, data, criterion, optimizer):
        self.train()
        
        total_loss = 0.0
        
        # Gabe: I'm using my custom batching here. using PyTorch's DataLoader might be faster, I haven't checked
        data.shuffle()
        for i in range(data.num_batches):
            inputs, labels = data.get_batch(i)
            
            optimizer.zero_grad()
        
            x, outputs = self(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        return total_loss / data.num_batches
    
    def test(self, data, criterion):
        self.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        inputs = torch.from_numpy(data.x).to(device)
        labels = torch.from_numpy(data.y).to(device)
        
        with torch.no_grad():
            x, outputs = self(inputs)
            loss = criterion(outputs, labels)
                
        return loss.item()

    def getMetrics(self, data):
        inputs = torch.from_numpy(data.x)
        labels = torch.from_numpy(data.y)

        
        with torch.no_grad():
            x, outputs = self(inputs)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()

        yTrue = labels
        yEst = predicted

        metricslist = []
        confusion_matrix = skl.confusion_matrix(yTrue,yEst) #make sure that this prints as array/list size [0,1,2,3]
        precision = skl.precision_score(yTrue, yEst)
        f1 = skl.f1_score(yTrue, yEst)
        recall = skl.recall_score(yTrue, yEst)
        accuracy = skl.accuracy_score(yTrue, yEst)
        brier_score = skl.brier_score_loss(confusion_matrix[0]+confusion_matrix[3],confusion_matrix[1]+confusion_matrix[3]) #first argument is tn + tp, second is fp + tp
        metricslist.append("accuracy = ", accuracy, "precision = ", precision, "recall = ", recall,"brier score= ", brier_score,"f1 score = ", f1)
        return metricslist
    
    def accuracy(self, data):
        self.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = torch.from_numpy(data.x).to(device)
        labels = torch.from_numpy(data.y).to(device)
        
        with torch.no_grad():
            x, outputs = self(inputs)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            
        return 100.0 * correct / labels.size(0)
    
    def accuracy_verbose(self, data, num_classes):
        self.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = torch.from_numpy(data.x).to(device)
        labels = torch.from_numpy(data.y).to(device)
        
        with torch.no_grad():
            x, outputs = self(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
        totals = np.zeros((num_classes))
        correct = np.zeros((num_classes))
        for i in range(labels.size(0)):
            totals[labels[i]] += 1
            if predicted[i] == labels[i]:
                correct[labels[i]] += 1
        
        accuracy = []
        for i in range(num_classes):
            accuracy.append(100.0 * correct[i] / totals[i])
        
        return accuracy

    