import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import os
import sys
import matplotlib.animation as animation
import matplotlib.pyplot as plt

def tensor_to_video(tensor, file_name):
    os.makedirs("videos", exist_ok=True)
    
    fig = plt.figure(frameon=False)
    ax = plt.axes([0, 0, 1, 1], frameon=False)
    ax.axis('off')
    img = ax.imshow(tensor[0].squeeze(), cmap='gray', vmin=0, vmax=1)

    def update(i):
        img.set_data(tensor[i].squeeze())
        return img,

    ani = animation.FuncAnimation(fig, update, frames=len(tensor), blit=True)
    ani.save(f"videos/{file_name}", writer='ffmpeg', fps=16, codec='mpeg2video')
    plt.close()

class Generator(nn.Module):
    def __init__(self, latent_dim, embedding_dim):
        super().__init__()
        
        self.z_proj = nn.Linear(latent_dim, 4*4*256)
        
        self.label_emb = nn.Embedding(10, embedding_dim)
        
        self.model = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(256, 128, 4, 2, 2, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 1, 1, 1, bias=False),
            nn.Tanh()
        )
        
    def forward(self, z, labels):
        
        batch_size = z.shape[0]
        
        embedded_labels = self.label_emb(labels).reshape(batch_size, 256, 4, 4)
        
        z = self.z_proj(z).reshape(batch_size, 256, 4, 4)
        
        z = torch.cat((z, embedded_labels), dim=1) # batch, 512, 4, 4
        
        return self.model(z)
    
    def generate_video(self, steps=32):
        z_original = torch.Tensor.repeat(torch.randn(1, 1, 100), steps, 1, 1)
        labels = torch.arange(10)
        embedded_labels = self.label_emb(labels)
        results = []
        
        batch_size = steps
        
        for i in range(10):
            z = z_original.clone()
            diff = (embedded_labels[(i+1)%10] - embedded_labels[i]) / steps
            embedded_diff_steps = torch.Tensor.repeat(embedded_labels[i], steps, 1, 1) + torch.Tensor.repeat(diff, steps, 1, 1) * torch.arange(steps).reshape(steps, 1, 1)
            embedded_diff_steps = embedded_diff_steps.reshape(batch_size, 256, 4, 4)
            z = self.z_proj(z).reshape(batch_size, 256, 4, 4)
            z = torch.cat((z, embedded_diff_steps), dim=1)
            results.append(self.model(z))
        
        results = torch.cat(results, dim=0)
        results = results.clamp(0, 1)
        return results.detach().numpy()

model = Generator(latent_dim=100, embedding_dim=4*4*256)
model.load_state_dict(torch.load("generator_state.pt", weights_only=True))

if __name__ == "__main__":
    
    if len(sys.argv) != 2:
        print("Usage: python generate.py <label>")
        sys.exit(1)

    try:
        label = int(sys.argv[1])
        if label < 0 or label > 9:
            raise ValueError("Label must be between 0 and 9")
    except ValueError as e:
        generated_images = model.generate_video()
        tensor_to_video(generated_images, sys.argv[1])
        sys.exit(1)
    
    model.eval()
    
    num_samples = 10
    
    with torch.no_grad():
        z = torch.randn(num_samples, 1, 100)
        labels = torch.tensor([label] * num_samples)
        img = model(z, labels)
        print(img.shape)
        
        grid = torchvision.utils.make_grid(img, nrow=5)
        plt.figure(figsize=(10, 10))
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap='gray')
        plt.axis('off')
        plt.show()