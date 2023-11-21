import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class LyricsDataset(Dataset):
    def __init__(self, lyrics, seq_length):
        self.lyrics = lyrics
        self.seq_length = seq_length

    def __len__(self):
        return len(self.lyrics) - self.seq_length

    def __getitem__(self, index):
        return (self.lyrics[index:index+self.seq_length],
                self.lyrics[index+1:index+self.seq_length+1])


def load_and_preprocess_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        data = file.read()
    data = data.lower().split()
    label_encoder = LabelEncoder()
    encoded_data = label_encoder.fit_transform(data)
    return encoded_data


class LSTMNet(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(LSTMNet, self).__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim,
                            num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out)
        return out


def train(model, criterion, optimizer, data_loader, num_epochs=10):
    for epoch in range(num_epochs):
        for inputs, targets in data_loader:
            inputs = inputs.clone().detach().to(torch.long)
            targets = targets.clone().detach().to(torch.long)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(
                outputs.view(-1, model.vocab_size), targets.view(-1))
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')


def main():
    data = load_and_preprocess_data('preprocessed_lyrics.txt')
    seq_length = 5
    dataset = LyricsDataset(data, seq_length)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    vocab_size = len(set(data))
    model = LSTMNet(vocab_size, embedding_dim=128,
                    hidden_dim=256, num_layers=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train(model, criterion, optimizer, data_loader)


if __name__ == '__main__':
    main()
