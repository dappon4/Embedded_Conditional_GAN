from model import EmbeddedConditionlGAN
from datamodule import MNISTDataModule
import pytorch_lightning as pl

if __name__ == "__main__":
    dm = MNISTDataModule(batch_size=256)
    model = EmbeddedConditionlGAN(img_shape=dm.dims, lr=0.005, b1=0.5, b2=0.999)
    trainer = pl.Trainer(max_epochs=500,accelerator="gpu")
    
    trainer.fit(model, dm)