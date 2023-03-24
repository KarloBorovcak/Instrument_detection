from intrument_model import InstrumentClassification
from instrument_dataset import InstrumentDataModule
import pytorch_lightning as pl
from torch import cuda
import config


if __name__ == "__main__":
    model = InstrumentClassification(num_classes=config.NUM_CLASSES,
                                     learning_rate=config.LEARNING_RATE,
                                     treshold=config.TRESHOLD)
    
    trainer = pl.Trainer(accelerator="gpu" if not cuda.is_available() else "cpu",
                         min_epochs=config.MIN_EPOCHS,
                         max_epochs=config.MAX_EPOCHS)
    
    dm = InstrumentDataModule(batch_size=config.BATCH_SIZE,
                              data_path=config.DATA_PATH,)
    
    # trainer.tune(model, dm) hyperparameter tuning
    trainer.fit(model, dm)
    trainer.validate(model, dm)
    trainer.test(model, dm)