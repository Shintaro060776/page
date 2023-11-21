import os
import json
import torch
import torch.nn as nn


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


def load_vocab_data(model_dir):
    with open(os.path.join(model_dir, 'vocab_size.json'), 'r') as f:
        vocab_size = json.load(f)

    with open(os.path.join(model_dir, 'vocab_dict.json'), 'r') as f:
        vocab_dict = json.load(f)

    with open(os.path.join(model_dir, 'vocab_list.json'), 'r') as f:
        vocab_list = json.load(f)

    return vocab_size, vocab_dict, vocab_list


def model_fn(model_dir):
    try:
        vocab_size, vocab_dict, vocab_list = load_vocab_data(model_dir)
        model = LSTMNet(vocab_size, embedding_dim=128,
                        hidden_dim=256, num_layers=2)
        model_path = os.path.join(model_dir, 'model.pth')
        with open(model_path, 'rb') as f:
            model.load_state_dict(torch.load(f))
        return model, vocab_dict, vocab_list
    except Exception as e:
        raise RuntimeError("Error loading the model: {}".format(e))


def input_fn(request_body, request_content_type):
    if request_content_type == 'application/json':
        try:
            data = json.loads(request_body)
            return data
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON in request body")
    else:
        raise ValueError(
            "Unsupported content type: {}".format(request_content_type))


def predict_fn(input_data, model):
    try:
        model, vocab_dict, _ = model
        input_tensor = tokenize_and_convert_to_tensor(input_data, vocab_dict)

        model.eval()
        with torch.no_grad():
            output = model(input_tensor)

        return output
    except Exception as e:
        raise RuntimeError("Error during prediction: {}".format())


def tokenize_and_convert_to_tensor(text, vocab_dict):
    try:
        tokens = text.lower().split()

        token_ids = [vocab_dict.get(token, vocab_dict['<unk>'])
                     for token in tokens]

        tensor = torch.tensor([token_ids], dtype=torch.long)

        return tensor
    except Exception as e:
        raise RuntimeError("Error in tokenization: {}".format(e))


def convert_output_to_lyrics(output, vocab_list):
    _, predicted_indices = torch.max(output, 2)

    predicted_words = [vocab_list[idx] for idx in predicted_indices[0]]

    lyrics = ' '.join(predicted_words)

    return lyrics


def output_fn(prediction, response_content_type):
    if response_content_type == 'application/json':
        try:
            return json.dumps(prediction)
        except TypeError:
            raise ValueError("Cannot serialize the prediction to JSON")
    else:
        raise ValueError(
            "Unsupported response content type: {}".format(response_content_type))
