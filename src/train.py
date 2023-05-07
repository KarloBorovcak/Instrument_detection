from model.instrument_model import InstrumentClassification
from model.dualnet import DualNet
from model.densenet import DenseNet
from model.instrument_dataset import InstrumentDataModule
import pytorch_lightning as pl
from torch import cuda, onnx, randn
import config


if __name__ == "__main__":
    model = DenseNet(num_labels=config.NUM_LABELS,
                                     learning_rate=config.LEARNING_RATE)
    
    trainer = pl.Trainer(accelerator="gpu" if cuda.is_available() else "cpu",
                         min_epochs=config.MIN_EPOCHS,
                         max_epochs=config.MAX_EPOCHS,
                         callbacks=[pl.callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=True, mode='min')])
    
    dm = InstrumentDataModule(batch_size=config.BATCH_SIZE,
                              training_data_path=config.PREPROCESSED_DATA_PATH)
    
        
    trainer.fit(model, dm)
    trainer.validate(model, dm)
    dummy_input = randn(1, 1, 128, 44)
    onnx.export(model, dummy_input, "./model/models/model_densenet_121.onnx")
