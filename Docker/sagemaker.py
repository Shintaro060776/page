from sagemaker.pytorch import PyTorch
import sagemaker
import os
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd

class_labels = {
    'smiling': 0,
    'openmouth': 1,
    'standing': 2,
    'nosmile': 3,
    'angry': 4,
    'stare': 5,
    'smile': 6,
    'sitting': 7,
    'walking': 8,
    'running': 9,
    'jumping': 10,
    'dancing': 11,
    'seesomething': 12,
    'understand': 13,
    'protect': 14,
}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=0.01)

    parser.add_argument('--model-dir', type=str,
                        default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str,
                        default=os.environ['SM_CHANNEL_TRAINING'])

    return parser.parse_args()


class CustomDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        try:
            img_name = self.img_labels.iloc[idx, 0]
            label_str = self.img_labels.iloc[idx, 1]
            label = class_labels[label_str]
            img_path = os.path.join(self.img_dir, img_name)
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error at index {idx}: {e}")
            raise


def train(model, train_loader, epochs, criterion, optimizer, device):
    model.train()
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            inputs, labels = data, target
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(
                    f'Epoch [{epoch+1}/{epochs}], Batch: {batch_idx}, Loss: {loss.item()}')


if __name__ == '__main__':
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_data = CustomDataset(
        annotations_file=os.path.join(args.data_dir, 'foldername.csv'),
        img_dir=os.path.join(args.data_dir, 'rendered_256x256'),
        transform=transform
    )

    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True)

    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(256*256*3, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 15)
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)

    train(model, train_loader, args.epochs, criterion, optimizer, device)

    model_save_path = os.path.join(args.model_dir, 'model.pth')
    torch.save(model.state_dict(), model_save_path)


sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()

estimator = PyTorch(entry_point='train_model.py',
                    role='arn:aws:iam::715573459931:role/test1111',
                    framework_version='1.5.0',
                    train_instance_count=1,
                    train_instance_type='ml.t2.medium',
                    hyperparameters={
                        'epochs': 10,
                        'batch-size': 64,
                        'learning-rate': 0.01
                    })

train_data = 's3://vhrthrtyergtcere/rendered_256x256/'

estimator.fit({'training': train_data})
