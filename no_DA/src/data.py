import numpy as np
import torch

# This class is compatible with PyTorch's DataLoader, or it can handle batching itself
class Data():
    def __init__(self, x, y, batch_size):
        self.batch_size = batch_size
        self.length = y.shape[0]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # todo: normalize the images
        self.x = np.array(x, dtype= np.float32)
        
        self.y = np.array(y, dtype=np.longlong)
        
        self.indices = np.array(range(self.length))
        self.num_batches = int(np.floor( self.length / batch_size))
        
    def __getitem__(self, index):
        return torch.from_numpy(self.x[index, :, :, :]).to(self.device), torch.from_numpy(self.y[index]).to(self.device) #self.y[index, :] <= this was throwing me an out of bound error
    
    def __len__(self):
        return self.length
    
    def shuffle(self):
        np.random.shuffle(self.indices)
    
    def get_num_batches(self):
        return self.num_batches
        
    def get_batch(self, batch_index):
        inputs = np.empty((self.batch_size, self.x.shape[1], self.x.shape[2], self.x.shape[3]), dtype= np.float32)
        labels = np.empty((self.batch_size), dtype= np.longlong)
        for i in range(self.batch_size):
            index = self.batch_size*batch_index + i
            inputs[i, :, :, :] = self.x[self.indices[index], :, :, :]
            labels[i] = self.y[self.indices[index]]
            
        return torch.from_numpy(inputs).to(self.device), torch.from_numpy(labels).to(self.device)
        
        
        
