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
    def __init__(self, latent_dim, embedding_dim):
        super().__init__()
        
        self.z_proj = nn.Linear(latent_dim, 4*4*256)
        self.embedded_proj = nn.Linear(embedding_dim, 4*4*256)
        
        self.tc1 = nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False)
        self.tc2 = nn.ConvTranspose2d(256, 128, 4, 2, 2, bias=False)
        self.tc3 = nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False)
        
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.conv2d = nn.Conv2d(64, 1, 1, 1, bias=False)
        
        self.leaky_relu = nn.LeakyReLU(0.2)
        
    def forward(self, z, embedded_labels):
        
        batch_size = z.shape[0]
        
        z = self.z_proj(z).reshape(batch_size, 256, 4, 4)
        embedded_labels = self.embedded_proj(embedded_labels).reshape(batch_size, 256, 4, 4)
        
        z = torch.cat((z, embedded_labels), dim=1) # batch, 512, 4, 4
        
        z = self.tc1(z) # batch, 256, 8, 8
        z = self.bn1(z)
        z = self.leaky_relu(z)

        z = self.tc2(z) # batch, 128, 14, 14
        z = self.bn2(z)
        z = self.leaky_relu(z)

        z = self.tc3(z) # batch, 64, 28, 28
        z = self.bn3(z)
        z = self.leaky_relu(z)

        z = self.conv2d(z)
        z = self.leaky_relu(z)
        z = torch.tanh(z)
        
        return z

class Discriminator(nn.Module):
    def __init__(self, img_shape, embedding_dim):
        super().__init__()
        
        conv_dim_1 = 16
        conv_dim_2 = 64
        
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()
        
        self.conv1 = nn.Conv2d(1+1, conv_dim_1, 4, 2, 1)
        self.conv2 = nn.Conv2d(conv_dim_1, conv_dim_2, 4, 2, 1)
        
        self.linear1 = nn.Linear(conv_dim_2*7*7, 128)
        self.linear2 = nn.Linear(128, 1)
        
        self.embedded_linear = nn.Linear(embedding_dim, np.prod(img_shape[1:]))
        
    def forward(self, img, embedded_labels):
        
        batch_size = img.shape[0]
        
        embedded_labels = self.embedded_linear(embedded_labels).reshape(batch_size, 1, 28, 28)
        
        img = self.leaky_relu(img)
        img = torch.cat((img, embedded_labels), dim=1) # batch, 2, 28, 28

        img = self.conv1(img) # batch, 64, 14, 14
        img = self.leaky_relu(img)
        img = self.dropout(img)
        
        img = self.conv2(img) # batch, 128, 7, 7
        img = self.leaky_relu(img)
        img = self.dropout(img)
        
        img = img.view(batch_size, -1)
        img = self.leaky_relu(self.linear1(img))
        img = self.linear2(img)
        
        #not needed for wasserstein loss
        img = self.sigmoid(img)

        return img

class Discriminator_2(nn.Module):
    def __init__(self, img_shape, embedding_dim):
        super().__init__()
        
        self.embedded_linear = nn.Linear(embedding_dim, np.prod(img_shape[1:]))
        
        self.model = nn.Sequential(
            nn.Conv2d(1+1, 1, 1, 1),
            nn.Flatten(),
            nn.LeakyReLU(0.2),
            nn.Linear(28*28, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img, embedded_labels):
        batch_size = img.shape[0]
        
        embedded_labels = self.embedded_linear(embedded_labels).reshape(batch_size, 1, 28, 28)
        
        img = torch.cat((img, embedded_labels), dim=1)
        
        return self.model(img)

class EmbeddedConditionlGAN(pl.LightningModule):
    def __init__(self, img_shape, g_lr, d_lr, b1, b2, n_critics=5, embedding_dim=100, latent_dim=100, label_smoothing_factor=0.1, gradient_clipping_value=0.5):
        super().__init__()
        self.save_hyperparameters()
        
        self.img_shape = img_shape
        self.g_lr = g_lr
        self.d_lr = d_lr
        self.b1 = b1
        self.b2 = b2
        self.n_critics = n_critics
        self.label_smoothing_factor = label_smoothing_factor
        self.gradient_clipping_value = gradient_clipping_value
        
        self.generator = Generator(latent_dim, embedding_dim)
        #self.discriminator = Discriminator(img_shape, embedding_dim)
        self.discriminator = Discriminator_2(img_shape, embedding_dim)
        
        self.emb = nn.Embedding(10, embedding_dim)
        
        self.leaky_relu = nn.LeakyReLU(0.2)

        # for validation
        self.latent_dim = latent_dim
        
        # model config
        self.automatic_optimization = False
        
    def forward(self, z, embedded_labels):
        return self.generator(z, embedded_labels)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)
    
    def wasserstein_loss(self, y_hat, y):
        return -torch.mean(y_hat * y)
    
    def training_step(self, batch):
        imgs, labels = batch
        
        ################
        # checking the data
        #print(labels)
        #grid = torchvision.utils.make_grid(imgs, nrow=5)
        #plt.figure(figsize=(10, 10))
        #plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap='gray')
        #plt.axis('off')
        #plt.show()
        ################
        
        optimizer_g, optimizer_d = self.optimizers()
        
        batch_size = imgs.shape[0]
        

        embedded_labels = self.emb(labels)
        
        # train generator
        self.toggle_optimizer(optimizer_g)
    
        z = torch.randn(batch_size, self.hparams.latent_dim).to(self.device)
        
        self.generated_imgs = self(z, embedded_labels)
        
        valid = torch.ones(imgs.size(0), 1) - self.hparams.label_smoothing_factor
        valid = valid.type_as(imgs)
        
        g_loss = self.adversarial_loss(self.discriminator(self.generated_imgs, embedded_labels), valid)
        
        self.manual_backward(g_loss, retain_graph=True)
        #self.manual_backward(g_loss)
        if self.hparams.gradient_clipping_value > 0:
            self.clip_gradients(optimizer_g, gradient_clip_val=self.gradient_clipping_value, gradient_clip_algorithm='norm')
        optimizer_g.step()
        optimizer_g.zero_grad()
        
        
        self.untoggle_optimizer(optimizer_g)
        
        # train discriminator
        #self.toggle_optimizer(optimizer_d)
        
        for _ in range(self.n_critics):
            
            self.toggle_optimizer(optimizer_d)
            
            valid = torch.ones(imgs.size(0), 1) * (1 - self.hparams.label_smoothing_factor)
            valid = valid.type_as(imgs)
            
            fake = torch.zeros(imgs.size(0), 1) + self.hparams.label_smoothing_factor
            fake = fake.type_as(imgs)

            real_loss = self.adversarial_loss(self.discriminator(imgs, embedded_labels), valid)
            #real_loss = self.wasserstein_loss(self.discriminator(imgs, embedded_labels), valid)

            fake_loss = self.adversarial_loss(self.discriminator(self.generated_imgs.detach(), embedded_labels), fake)
            
            #d_loss = (real_loss + fake_loss) / 2
            d_loss = real_loss + fake_loss
            
            self.manual_backward(d_loss, retain_graph=True)
            if self.hparams.gradient_clipping_value > 0:
                self.clip_gradients(optimizer_d, gradient_clip_val=self.gradient_clipping_value, gradient_clip_algorithm='norm')
            optimizer_d.step()
            optimizer_d.zero_grad()
        
            self.untoggle_optimizer(optimizer_d)
        
        self.log('g_loss', g_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('d_loss', d_loss, on_step=True, on_epoch=True, prog_bar=True)
        
    def configure_optimizers(self):
        g_lr = self.hparams.g_lr
        d_lr = self.hparams.d_lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2
        
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=g_lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=d_lr, betas=(b1, b2))
        #opt_d = torch.optim.RMSprop(self.discriminator.parameters(), lr=d_lr)
        
        return [opt_g, opt_d], []

    def validation_step(self, batch):
        pass
    
    def on_validation_epoch_end(self):
        if (self.current_epoch + 1) % 5 == 0:
            
            validation_z = torch.randn(10, self.latent_dim)
            validation_labels = torch.arange(0, 10)
            
            generated_imgs = self(validation_z.to(self.device), self.emb(validation_labels.to(self.device)))
            grid = torchvision.utils.make_grid(generated_imgs, nrow=5)
            
            plt.figure(figsize=(10, 10))
            plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap='gray')
            plt.axis('off')
            plt.show()