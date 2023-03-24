from torch import nn, optim, save, load
import pytorch_lightning as pl
import torchmetrics as metrics


class InstrumentClassification(pl.LightningModule):
    def __init__(self, num_classes, learning_rate, treshold):
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
        self.accuracy = metrics.Accuracy(task="multilabel", num_classes=num_classes, threshold=treshold)
        self.loss = nn.BCELoss()
        self.cnt = 0
        self.avg_loss = 0
        self.learning_rate = learning_rate

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


    def _common_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        return loss, y_hat, y
    
    def training_step(self, batch, batch_idx):
        loss, y_hat, y = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(y_hat, y)
        self.log_dict({'train_loss': loss, 'train_acc': accuracy},
                      prog_bar=True, logger=True, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, y_hat, y = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(y_hat, y)
        self.log_dict({'val_loss': loss, 'val_acc': accuracy},
                      prog_bar=True, logger=True, on_step=False, on_epoch=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss, y_hat, y = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(y_hat, y)
        self.log_dict({'test_loss': loss, 'test_acc': accuracy},
                      prog_bar=True, logger=True, on_step=False, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)
    
    def save(self, path):
        save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(load(path))

