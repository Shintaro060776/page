import argparse
import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer


class JokesDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


class JokeGeneratorModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        logits = self.fc(lstm_out)
        return logits


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--vocabulary-size', type=int, default=10000)
    parser.add_argument('--data-dir', type=str,
                        default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--model-dir', type=str,
                        default=os.environ['SM_MODEL_DIR'])

    args = parser.parse_known_args()[0]

    df = pd.read_csv(os.path.join(args.data_dir, 'normalized_jokes.csv'))
    jokes = df['Normalized Joke'].tolist()

    token_counts = Counter(word for joke in jokes for word in joke.split())
    vocab = {word: i + 1 for i,
             (word, _) in enumerate(token_counts.most_common(args.vocabulary_size - 1))}
    vocab['<PAD>'] = 0

    encoded_jokes = [[vocab[word] for word in joke.split() if word in vocab]
                     for joke in jokes]

    filtered_encoded_jokes = [seq for seq in encoded_jokes if len(seq) >= 2]

    if not filtered_encoded_jokes:
        raise ValueError(
            "All sequences are shorter than the minimum length after filtering.")

    sequences = pad_sequence([torch.tensor(seq[:-1]) for seq in filtered_encoded_jokes],
                             padding_value=vocab['<PAD>'], batch_first=True)
    labels = pad_sequence([torch.tensor(seq[1:]) for seq in filtered_encoded_jokes],
                          padding_value=vocab['<PAD>'], batch_first=True)

    dataset = JokesDataset(sequences, labels)
    train_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True)

    model = JokeGeneratorModel(
        args.vocabulary_size, embed_dim=100, hidden_dim=128)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters())

    for epoch in range(args.epochs):
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(
                outputs.view(-1, args.vocabulary_size), targets.view(-1))
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{args.epochs} Loss: {loss.item()}')

    torch.save(model.state_dict(), os.path.join(
        args.model_dir, 'model.pth'))
