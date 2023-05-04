from torch import nn, optim
import pytorch_lightning as pl
import torchmetrics as metrics


class InstrumentClassification(pl.LightningModule):
    def __init__(self, num_labels, learning_rate):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3))
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3))
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3))
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(),
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((1, 1))
        )



        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_labels)
        self.dropout2 = nn.Dropout(0.25)
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        self.accuracy = metrics.Accuracy(task="multilabel", num_labels=num_labels)
        self.f1 = metrics.F1Score(task="multilabel", num_labels=num_labels)
        self.loss = nn.BCELoss()
        self.cnt = 0
        self.avg_loss = 0
        self.learning_rate = learning_rate

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.dropout2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.dropout2(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.dropout2(x)
        x = self.conv7(x)
        x = self.conv8(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
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
    
