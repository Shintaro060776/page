import os
import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam

from wavenet_vocoder import WaveNet
from wavenet_vocoder.datasets import WaveNetDataset

epochs = 100
learning_rate = 0.001
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = WaveNetDataset(data_path=args.train)
val_dataset = WaveNetDataset(data_path=args.validation)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

model = WaveNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=learning_rate)


def train(args):
    model = WaveNet().to(device)
    optimizer = Adam(model.parameters(), lr=args["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args["epochs"]):
        model.train()
        total_train_loss = 0

        for batch in train_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        print(
            f"Epoch {epoch+1}/{args['epochs']}, Training Loss: {avg_train_loss}")

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        print(
            f"Epoch {epoch+1}/{args['epochs']}, Validation Loss: {avg_val_loss}")

        torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--model-dir', type=str, default='./model')
    parser.add_argument('--train', type=str, default="/data/train_data")
    parser.add_argument('--validation', type=str,
                        default="/data/validation_data")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    train_dataset = WaveNetDataset(data_path=args.train)
    val_dataset = WaveNetDataset(data_path=args.validation)
