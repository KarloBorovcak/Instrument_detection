import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split
from torch import from_numpy
import numpy as np

class InstrumentDataset(Dataset):
    def __init__(self, split, data_path):
        self.split = split
        self.data = np.load(f'{data_path}/{split}.npy')
        self.labels = np.load(f'{data_path}/{split}_labels.npy')
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

class InstrumentDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, data_path):
        super().__init__()
        self.batch_size = batch_size
        self.data_path = data_path
        
    def setup(self, stage=None):
        self.train_dataset = InstrumentDataset('train', self.data_path)
        test_dataset_temp = InstrumentDataset('test', self.data_path)
        size_test_val = len(test_dataset_temp)//2
        self.test_dataset, self.val_dataset = random_split(test_dataset_temp, [size_test_val, size_test_val])
        
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)



