import os
import json
import torch


class JokeGeneratorModel(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(JokeGeneratorModel, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim)
        self.lstm = torch.nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        logits = self.fc(lstm_out)
        return logits


def model_fn(model_dir):
    model = JokeGeneratorModel(vocab_size=10000, embed_dim=100, hidden_dim=128)
    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f))
    model.eval()
    return model


def input_fn(serialized_input_data, content_type):
    if content_type == 'application/json':
        input_data = json.loads(serialized_input_data)
        return torch.tensor(input_data)
    else:
        raise ValueError("Unsupported content type: {}".format(content_type))


def predict_fn(input_data, model):
    with torch.no_grad():
        output = model(input_data)
    return output


def output_fn(prediction_output, accept):
    if accept == 'application/json':
        return json.dumps(prediction_output.numpy().tolist()), accept
    else:
        raise ValueError("Unsupported accept type: {}".format(accept))
