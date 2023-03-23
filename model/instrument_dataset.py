from torch.utils.data import Dataset
from torch import from_numpy
import numpy as np

class InstrumentDataset(Dataset):
    def __init__(self, split):
        self.split = split
        self.data = np.load(f'./{split}.npy')
        self.labels = np.load(f'./{split}_labels.npy')
        # self.transform = transforms.Compose([
        #     transforms.ToTensor(),
        # ])
        
    def __len__(self):
        return len(self.data)
    
    def transform(self, x):
        return np.resize(x, (1, 128, 44))
    
    def tensorize(self, x):
        return from_numpy(np.ndarray.astype(x, np.float32))
    
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        x = self.transform(x)
        x = self.tensorize(x)
        y = self.tensorize(y)
        return x, y