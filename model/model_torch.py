from torch import nn, optim, save, load, stack
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from instrument_dataset import InstrumentDataset


class InstrumentClassification(pl.LightningModule):
    def __init__(self, num_classes=11):
        super().__init__()
        self.conv1 = nn.Conv2d(1 , 32, (3, 3))
        self.maxpool = nn.MaxPool2d((3, 3), (3, 3))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
        
        # Flatten layer
        self.flatten = nn.Flatten()
        
        # Dense layers
        self.fc1 = nn.Linear(3328, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        #self.accuracy = .metrics.Accuracy()
        self.loss = nn.BCELoss()
        self.cnt = 0
        self.avg_loss = 0

    def forward(self, x):
        # Convolutional layers
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Flatten layer
        x = self.flatten(x)
        
        # Dense layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        
        return x

    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        return loss
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)
    
    def train_dataloader(self):
        return DataLoader(InstrumentDataset('train'), batch_size=32, shuffle=True)
    
    # def val_dataloader(self):
    #     return DataLoader(InstrumentDataset('val'), batch_size=32, shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(InstrumentDataset('test'), batch_size=32, shuffle=False)
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        self.cnt += 1
        self.avg_loss = loss if self.cnt == 1 else (self.avg_loss * (self.cnt - 1) + loss) / self.cnt
        return loss
    
    def on_test_epoch_end(self):
        pass
        
    def on_validation_epoch_end(self):
        pass
    
    def on_train_epoch_end(self):
        tensorboard_logs = {'test_loss': self.loss}
        return {'test_loss': self.loss, 'log': tensorboard_logs}
    
    def save(self, path):
        save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(load(path))



if __name__ == "__main__":
    model = InstrumentClassification()
    trainer = pl.Trainer(accelerator="cpu", max_epochs=1)
    trainer.fit(model, model.train_dataloader())
    trainer.test(model, model.test_dataloader())
    model.save('model.pt')




    