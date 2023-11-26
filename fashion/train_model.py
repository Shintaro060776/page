import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from PIL import Image
from torch.utils.data import Dataset
import traceback
import boto3
import io


class S3CustomDataset(Dataset):
    def __init__(self, bucket, prefix, transform=None):
        self.s3_client = boto3.client('s3')
        self.bucket = bucket
        self.prefix = prefix
        self.transform = transform

        self.images = self.get_images_from_s3()

    def get_images_from_s3(self):
        response = self.s3_client.list_objects_v2(
            Bucket=self.bucket, Prefix=self.prefix)
        return [content['Key'] for content in response.get('Contents', []) if content['Key'].endswith('.jpg')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_key = self.images[idx]
        img_data = self.s3_client.get_object(Bucket=self.bucket, Key=img_key)
        img = Image.open(io.BytesIO(img_data['Body'].read()))

        if self.transform:
            img = self.transform(img)

        return img


class Generator(nn.Module):
    def __init__(self, nz):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(args.nz, 512, 4, 1, 0, bias=False),
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
            nn.ConvTranspose2d(64, 3, (4, 5), (2, 2), (1, 1), bias=False),
            # nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
            # nn.AdaptiveAvgPool2d((533, 400))
            nn.AdaptiveAvgPool2d((256, 256))
        )

    def forward(self, x):
        x = self.main(x)
        print("Generator output shape:", x.shape)
        return x


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
            nn.Flatten()
        )

        final_output_size = 13 * 13 * 1

        # self.linear = nn.Linear(660, 1)
        self.linear = nn.Linear(final_output_size, 1)
        self.output = nn.Sigmoid()

    def forward(self, x):
        x = self.main(x)
        print("Discriminator flattened output shape:", x.shape)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        print("Discriminator final output shape:", x.shape)
        return self.output(x)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((533, 400)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    bucket, prefix = args.data_dir.replace('s3://', '').split('/', 1)

    train_dataset = S3CustomDataset(
        bucket=bucket, prefix=prefix, transform=transform)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True)

    netG = Generator(args.nz).to(device)
    netD = Discriminator().to(device)
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=args.learning_rate)
    optimizerG = optim.Adam(netG.parameters(), lr=args.learning_rate)

    fixed_noise = torch.randn(64, args.nz, 1, 1, device=device)

    for epoch in range(args.epochs):
        for i, images in enumerate(train_loader):
            images = images.to(device)
            print(f"Processing epoch {epoch+1}, batch {i+1}")
            print("Batch loaded")
            try:
                print(f"Batch {i}, Image shape: {images.shape}")

                netD.zero_grad()
                real_labels = torch.ones(images.size(0), 1).to(device)
                outputs = netD(images).view(-1, 1)
                print("Output shape (real):", outputs.shape)
                print("Target shape (real):", real_labels.shape)
                d_loss_real = criterion(outputs, real_labels)
                d_loss_real.backward()

                print("Starting discriminator training")
                noise = torch.randn(images.size(0), args.nz, 1, 1)
                fake_images = netG(noise)
                fake_labels = torch.zeros(images.size(0), 1).to(device)
                outputs = netD(fake_images.detach()).view(-1, 1)
                print("Output shape (fake):", outputs.shape)
                print("Target shape (fake:)", fake_labels.shape)
                d_loss_fake = criterion(outputs, fake_labels)
                d_loss_fake.backward()
                optimizerD.step()

                print("Starting generator training")
                netG.zero_grad()
                outputs = netD(fake_images)
                g_loss = criterion(outputs, real_labels)
                g_loss.backward()
                optimizerG.step()

                output_dir = '/opt/ml/model/fashion/sample'

                print(
                    f'Epoch [{epoch+1}/{args.epochs}], Step [{i+1}/{len(train_loader)}], d_loss: {d_loss_real.item() + d_loss_fake.item()}, g_loss: {g_loss.item()}')

                if i == len(train_loader) - 1:
                    os.makedirs(output_dir, exist_ok=True)
                    save_path = os.path.join(
                        output_dir, f'fake_images-{epoch+1}.png')
                    save_image(fake_images, save_path, nrow=8, normalize=True)

            except Exception as e:
                print(
                    f"Error during training at epoch {epoch+1}, batch {i+1}: {e}")
                traceback.print_exc()
                return

        if (epoch+1) % 10 == 0:
            fake_images = netG(fixed_noise)

    torch.save(netG.state_dict(), os.path.join(
        args.model_dir, 'model.pth'))
    torch.save(netD.state_dict(), os.path.join(
        args.model_dir, 'discriminator.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.0002)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--model_dir', type=str, default='/opt/ml/model')
    parser.add_argument('--data_dir', type=str, default='/opt/ml/input/data')

    args = parser.parse_args()
    train(args)
