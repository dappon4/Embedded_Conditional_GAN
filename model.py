import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision.utils import save_image
from torch import Tensor
import matplotlib.pyplot as plt

from datamodule import MNISTDataModule

class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()

        self.latent_dim = latent_dim
        
        self.leaky_relu = nn.LeakyReLU(0.2)
        
        self.lienar1 = nn.Linear(latent_dim, 256)
        self.linear2 = nn.Linear(256, 7*7*16)
        
        self.tc1 = nn.ConvTranspose2d(16, 64, 4, 2, 1)
        self.tc2 = nn.ConvTranspose2d(64, 1, 4, 2, 1)
        
    def forward(self, z):
        z = self.leaky_relu(self.lienar1(z))
        z = self.leaky_relu(self.linear2(z))
        
        z = z.reshape(z.shape[0], 16, 7, 7)
        
        z = self.tc1(z) # batch, 64, 14, 14
        z = self.leaky_relu(z)
        z = self.tc2(z)
        z = torch.tanh(z)

        return z

class Disciminator(nn.Module):
    def __init__(self, img_shape):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity

class EmbeddedConditionlGAN(pl.LightningModule):
    def __init__(self, img_shape, lr, b1, b2, latent_dim=100):
        super().__init__()
        self.save_hyperparameters()
        
        self.img_shape = img_shape
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        
        self.generator = Generator()
        self.discriminator = Disciminator(img_shape=img_shape)
        
        self.emb = nn.Embedding(10, latent_dim)
        self.d_emb_linear = nn.Linear(latent_dim, np.prod(img_shape))
        
        self.leaky_relu = nn.LeakyReLU(0.2)

        # for validation
        self.validation_z = torch.randn(10, latent_dim)
        self.validation_labels = torch.arange(0, 10)
        
        # model config
        self.automatic_optimization = False
        
    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)
    
    def training_step(self, batch):
        imgs, labels = batch
        
        optimizer_g, optimizer_d = self.optimizers()
        
        batch_size = imgs.shape[0]
        z = torch.randn(batch_size, self.hparams.latent_dim).to(self.device)
        z.type_as(imgs)

        embedded_labels = self.emb(labels)
        discriminator_embedded_labels = self.d_emb_linear(embedded_labels).reshape(batch_size, *self.img_shape)
        
        # train generator
        self.toggle_optimizer(optimizer_g)
        self.generated_imgs = self(z + embedded_labels)
        
        valid = torch.ones(imgs.size(0), 1)
        valid = valid.type_as(imgs)
        
        g_loss = self.adversarial_loss(self.discriminator(self.generated_imgs + discriminator_embedded_labels), valid)
        
        self.manual_backward(g_loss, retain_graph=True)
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)
        
        # train discriminator
        self.toggle_optimizer(optimizer_d)
        
        valid = torch.ones(imgs.size(0), 1)
        valid = valid.type_as(imgs)

        real_loss = self.adversarial_loss(self.discriminator(imgs + discriminator_embedded_labels), valid)

        fake = torch.zeros(imgs.size(0), 1)
        fake = fake.type_as(imgs)

        fake_loss = self.adversarial_loss(self.discriminator(self.generated_imgs.detach() + discriminator_embedded_labels), fake)
        
        d_loss = (real_loss + fake_loss) / 2
        self.manual_backward(d_loss)
        optimizer_d.step()
        optimizer_d.zero_grad()
        self.untoggle_optimizer(optimizer_d)
        
        
    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2
        
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        
        return [opt_g, opt_d], []

    def validation_step(self, batch):
        pass
    
    def on_validation_epoch_end(self):
        if (self.current_epoch + 1) % 5 == 0:
            generated_imgs = self(self.validation_z.to(self.device) + self.emb(self.validation_labels.to(self.device)))
            grid = torchvision.utils.make_grid(generated_imgs, nrow=5)
            
            plt.figure(figsize=(10, 10))
            plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap='gray')
            plt.axis('off')
            plt.show()
    
    
if __name__ == "__main__":
    dm = MNISTDataModule()
    dm.prepare_data()
    dm.setup()
    
    gen = Generator(dm.dims)
    gen(torch.randn(1, 100), Tensor([1]).long())