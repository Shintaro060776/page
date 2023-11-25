import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from PIL import Image
from torch.utils.data import Dataset
import os
import traceback

batch_size = 64
learning_rate = 0.0002
epochs = 50
nz = 100


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = [os.path.join(root_dir, fname) for fname in os.listdir(
            root_dir) if fname.endswith('.jpg')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        return image


transform = transforms.Compose([
    transforms.Resize((533, 400)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = CustomDataset(
    root_dir='/page/fashion/images', transform=transform)
print(f"Number of images in dataset: {len(train_dataset)}")
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Flatten(),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.main(x)
        return x.view(-1)


def train(netG, netD, criterion, optimizerG, optimizerD):
    print("Training started")
    for epoch in range(epochs):
        for i, images in enumerate(train_loader):
            print(f"Processing epoch {epoch+1}, batch {i+1}")
            print("Batch loaded")
            try:
                print(f"Batch {i}, Image shape: {images.shape}")

                netD.zero_grad()
                real_labels = torch.ones(images.size(0))
                outputs = netD(images)
                d_loss_real = criterion(outputs, real_labels)
                d_loss_real.backward()

                print("Starting discriminator training")
                noise = torch.randn(images.size(0), nz, 1, 1)
                fake_images = netG(noise)
                fake_labels = torch.zeros(images.size(0))
                outputs = netD(fake_images.detach())
                d_loss_fake = criterion(outputs, fake_labels)
                d_loss_fake.backward()
                optimizerD.step()

                print("Starting generator training")
                netG.zero_grad()
                outputs = netD(fake_images)
                g_loss = criterion(outputs, real_labels)
                g_loss.backward()
                optimizerG.step()

                if (i+1) % 100 == 0:
                    print(
                        f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], d_loss: {d_loss_real.item() + d_loss_fake.item()}, g_loss: {g_loss.item()}')

            except Exception as e:
                print(
                    f"Error during training at epoch {epoch+1}, batch {i+1}: {e}")
                traceback.print_exc()
                continue

        if (epoch+1) % 10 == 0:
            fake_images = netG(fixed_noise)
            save_image(fake_images, os.path.join('/page/fashion/sample',
                       f'fake_images-{epoch+1}.png'), nrow=8, normalize=True)

    torch.save(netG.state_dict(), './generator.pth')
    torch.save(netD.state_dict(), './discriminator.pth')


if __name__ == '__main__':

    netG = Generator()
    netD = Discriminator()
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=learning_rate)
    optimizerG = optim.Adam(netG.parameters(), lr=learning_rate)

    fixed_noise = torch.randn(64, nz, 1, 1)

    train(netG, netD, criterion, optimizerG, optimizerD)
