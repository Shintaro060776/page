import os
import torch
import torch.nn as nn
import json
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(filename='inference.log', level=logging.DEBUG)


class Generator(nn.Module):
    def __init__(self, nz):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 3, (4, 5), (2, 2), (1, 1), bias=False),
            nn.Tanh(),
            nn.AdaptiveAvgPool2d((256, 256))
        )

    def forward(self, x):
        return self.main(x)


def model_fn(model_dir):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Generator(nz=100)
        with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
            model.load_state_dict(torch.load(f, map_location=device))
        return model.to(device)
    except Exception as e:
        logger.error("Error in model_fn: {}".format(e))
        raise


def input_fn(request_body, request_content_type):
    try:
        if request_content_type == 'application/json':
            data = json.loads(request_body)
            return data
        else:
            raise ValueError(
                "Unsupported content type: {}".format(request_content_type))
    except Exception as e:
        logger.error("Error in input_fn: {}".format(e))
        raise


def predict_fn(input_data, model):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()

        nz = 100

        if 'nz' in input_data:
            nz = input_data['nz']
        else:
            nz = 100

        with torch.no_grad():
            noise = torch.randn(1, nz, 1, 1, device=device)
            return model(noise)
    except Exception as e:
        logger.error("Error in predict_fn: {}".format(e))
        raise


def output_fn(prediction, content_type):
    try:
        if content_type == 'application/json':
            return json.dumps(prediction.cpu().numpy().tolist())
        else:
            raise ValueError(
                "Unsupported content type: {}".format(content_type))
    except Exception as e:
        logger.error("Error in output_fn: {}".format(e))
        raise
