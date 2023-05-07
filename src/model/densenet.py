from torch import nn, optim, save, load
import pytorch_lightning as pl
import torchmetrics as metrics
from torchvision import models


class DenseNet(pl.LightningModule):
    def __init__(self, num_labels, learning_rate):
        super().__init__()
        
        preloaded = models.densenet121()
        self.features = preloaded.features
        self.features.conv0 = nn.Conv2d(1, 64, 7, 2, 3)
        self.classifier = nn.Linear(1024, num_labels, bias=True)
        
        del preloaded
        
        self.sigmoid = nn.Sigmoid()
        self.accuracy = metrics.Accuracy(task="multilabel", num_labels=num_labels)
        self.f1 = metrics.F1Score(task="multilabel", num_labels=num_labels)
        self.loss = nn.BCELoss()
        self.cnt = 0
        self.avg_loss = 0
        self.learning_rate = learning_rate

    def forward(self, x):
        features = self.features(x)
        out = nn.functional.relu(features, inplace=True)
        out = nn.functional.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        out = self.classifier(out)
        out = self.sigmoid(out)
        
        return out
        



    def _common_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        return loss, y_hat, y
    
    def training_step(self, batch, batch_idx):
        loss, y_hat, y = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(y_hat, y)
        f1 = self.f1(y_hat, y)
        self.log_dict({'train_loss': loss, 'train_acc': accuracy, 'train_f1': f1},
                      prog_bar=True, logger=True, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, y_hat, y = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(y_hat, y)
        f1 = self.f1(y_hat, y)
        self.log_dict({'val_loss': loss, 'val_acc': accuracy, 'val_f1': f1},
                      prog_bar=True, logger=True, on_step=False, on_epoch=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss, y_hat, y = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(y_hat, y)
        f1 = self.f1(y_hat, y)
        self.log_dict({'test_loss': loss, 'test_acc': accuracy, 'test_f1': f1},
                      prog_bar=True, logger=True, on_step=False, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)
    
    def save(self, path):
        save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(load(path))
