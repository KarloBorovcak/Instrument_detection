import pytorch_lightning as pl
import config
from torch.utils.data import Dataset, DataLoader, random_split
from torch import from_numpy
import numpy as np
import os


class InstrumentTrainingDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.instrument_list = config.INSTRUMENTS
        self.file_list = []
        self.label_list = []

        for instrument in self.instrument_list:
            data_path_instrument = os.path.join(data_path, instrument)
            temp_files = os.listdir(data_path_instrument)
            for file in temp_files:
                self.file_list.append(os.path.join(data_path_instrument, file))
            temp = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            temp[self.instrument_list.index(instrument)] = 1
            self.label_list += [temp] * len(temp_files)
        
        
    def __len__(self):
        return len(self.file_list)
    
    def transform(self, x):
        return np.resize(x, (1, 128, 44))
    
    def tensorize(self, x):
        return from_numpy(x)
    
    def __getitem__(self, idx):
        x = np.load(self.file_list[idx])
        y = self.label_list[idx]
        x = self.transform(x)
        x = self.tensorize(np.ndarray.astype(np.array(x), np.float32))
        y = self.tensorize(np.ndarray.astype(np.array(y), np.float32))
        return x, y
    

class InstrumentDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, training_data_path):
        super().__init__()
        self.batch_size = batch_size
        self.data_path = training_data_path
        
    def setup(self, stage=None):
        dataset = InstrumentTrainingDataset(self.data_path)
        self.train_dataset, self.val_dataset = random_split(dataset, [dataset.__len__() - int(dataset.__len__() * config.VALIDATION_SPLIT), int(dataset.__len__() * config.VALIDATION_SPLIT)])
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=config.NUM_WORKERS)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=config.NUM_WORKERS)
