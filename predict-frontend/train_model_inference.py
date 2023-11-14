import os
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import json
import base64
import io
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

    if 'SM_MODEL_DIR' not in os.environ:
        os.environ['SM_MODEL_DIR'] = './'
    if 'SM_CHANNEL_TRAINING' not in os.environ:
        os.environ['SM_CHANNEL_TRAINING'] = './'

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


def input_fn(request_body, request_content_type):
    try:
        if request_content_type == 'application/json':
            input_data = json.loads(request_body)
            base64_image = input_data['instances'][0]
            decoded_image = base64.b64decode(base64_image)

            image = Image.open(io.BytesIO(decoded_image)).convert('RGB')

            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            image_tensor = transform(image).unsqueeze(0)

            return image_tensor
        else:
            raise ValueError(
                f"Unsupported content type: {request_content_type}")
    except Exception as e:
        logger.error(f"input_fn processing error: {e}")
        raise


def predict_fn(input_data, model):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        input_data = input_data.to(device)

        with torch.no_grad():
            model.eval()
            output = model(input_data)

        return output
    except Exception as e:
        logger.error(f"predict_fn processing error: {e}")
        raise


def output_fn(prediction_output, accept):
    try:
        if accept == 'application/json':
            response = prediction_output.cpu().numpy().tolist()
            return json.dumps(response), accept
        else:
            raise ValueError(f"Unsupported accept type: {accept}")
    except Exception as e:
        logger.error(f"output_fn processing error: {e}")
        raise


def model_fn(model_dir):
    try:
        model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(256*256*3, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 15)
        )

        model_path = os.path.join(model_dir, 'model.pth')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        model.load_state_dict(torch.load(model_path))
        return model
    except Exception as e:
        logger.error(f"model_fn loading error: {e}")
        raise


if __name__ == '__main__':
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
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
