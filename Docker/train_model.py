import os
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
        # Assuming label is already an integer. If not, convert it here.
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error at index {idx}: {e}")
            raise


# Define a transform to normalize the data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256)),
    transforms.Normalize((0.5,), (0.5,))
])

# Assuming folder structure is '/page/Docker/rendered_256x256/cat' and '/page/Docker/rendered_256x256/human'
train_data = CustomDataset(
    annotations_file='foldername.csv',
    img_dir='rendered_256x256',
    transform=transform
)

train_loader = DataLoader(train_data, batch_size=4, shuffle=True)

# Define your model (Example using a simple model, replace with your actual model architecture)
model = torch.nn.Sequential(
    torch.nn.Flatten(),
    torch.nn.Linear(256*256*3, 512),
    torch.nn.ReLU(),
    torch.nn.Linear(512, 15)  # Assuming two classes: cat and human
)

# Loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    for i, data in enumerate(train_loader):
        try:
            inputs, labels = data
            # Check the shape of inputs and labels
            print(
                f"Batch {i}, inputs shape: {inputs.shape}, labels shape: {labels.shape}")

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')
        except Exception as e:
            print(f"Error in batch {i}: {e}")
            # エラーがあればそれ以上実行しない
            break

# Save the model
model_save_path = '/page/Docker/model.pth'
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
torch.save(model.state_dict(), model_save_path)
