import argparse
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--seq_length', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.001)

    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--data_dir', type=str,
                        default=os.environ['SM_CHANNEL_TRAIN'])

    return parser.parse_args()


class LSTMNet(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(LSTMNet, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim,
                            num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out)
        return out


class LyricsDataset(Dataset):
    def __init__(self, lyrics, seq_length):
        self.lyrics = lyrics
        self.seq_length = seq_length

    def __len__(self):
        return len(self.lyrics) - self.seq_length

    def __getitem__(self, index):
        inputs = torch.tensor(
            self.lyrics[index:index+self.seq_length], dtype=torch.long)
        targets = torch.tensor(
            self.lyrics[index+1:index+self.seq_length+1], dtype=torch.long)
        return inputs, targets


def load_and_preprocess_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        text = file.read()

    text = text.lower().split()
    vocab = set(text)
    vocab_size = len(vocab)

    word_to_index = {word: i for i, word in enumerate(vocab)}
    index_to_word = {i: word for i, word in enumerate(vocab)}

    encoded_text = [word_to_index[word] for word in text]

    return encoded_text, word_to_index, index_to_word


def main(args):
    data, vocab_dict, vocab_list = load_and_preprocess_data(
        os.path.join(args.data_dir, 'preprocessed_lyrics.txt'))
    dataset = LyricsDataset(data, args.seq_length)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    vocab_size = len(vocab_dict)
    model = LSTMNet(vocab_size, args.embedding_dim,
                    args.hidden_dim, args.num_layers)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        for inputs, targets in data_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.view(-1, vocab_size)
            loss = criterion(outputs, targets.view(-1))
            loss.backward()
            optimizer.step()

            print(f'Epoch {epoch+1}/{args.epochs}, Loss: {loss.item()}')

    if args.model_dir is None:
        model_dir = os.environ.get('SM_MODEL_DIR', '.')
    else:
        model_dir = args.model_dir

    model_dir = os.environ.get('SM_MODEL_DIR', args.model_dir)
    torch.save(model.state_dict(), os.path.join(model_dir, 'model.pth'))
    save_vocab_data(args.model_dir, vocab_size, vocab_dict, vocab_list)


def save_vocab_data(model_dir, vocab_size, vocab_dict, vocab_list):
    if model_dir is None:
        model_dir = os.environ.get('SM_MODEL_DIR', '.')

    with open(os.path.join(model_dir, 'vocab_size.json'), 'w') as f:
        json.dump(vocab_size, f)

    with open(os.path.join(model_dir, 'vocab_dict.json'), 'w') as f:
        json.dump(vocab_dict, f)

    with open(os.path.join(model_dir, 'vocab_list.json'), 'w') as f:
        json.dump(vocab_list, f)


def load_vocab_data(model_dir):
    with open(os.path.join(model_dir, 'vocab_size.json'), 'r') as f:
        vocab_size = json.load(f)

    with open(os.path.join(model_dir, 'vocab_dict.json'), 'r') as f:
        vocab_dict = json.load(f)

    with open(os.path.join(model_dir, 'vocab_list.json'), 'r') as f:
        vocab_list = json.load(f)

    return vocab_size, vocab_dict, vocab_list


if __name__ == '__main__':
    args = parse_args()
    main(args)
