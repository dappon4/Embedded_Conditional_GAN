from model import EmbeddedConditionlGAN
from datamodule import MNISTDataModule
import pytorch_lightning as pl
import torch

torch.set_float32_matmul_precision("high")

if __name__ == "__main__":
    dm = MNISTDataModule(batch_size=32)
    model = EmbeddedConditionlGAN(img_shape=dm.dims, g_lr=0.0003, d_lr=0.0001, b1=0.5, b2=0.999, label_smoothing_factor=0, n_critics=5, gradient_clipping_value=0)
    trainer = pl.Trainer(max_epochs=50,accelerator="gpu")
    
    trainer.fit(model, dm)
    torch.save(model.state_dict(), "model.pth")