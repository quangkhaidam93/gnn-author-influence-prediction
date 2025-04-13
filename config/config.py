from datetime import datetime

import torch


class Config:
    INFLUENCE_THRESHOLD = 10
    HIDDEN_CHANNELS = 30
    NUM_HEADS = 2
    LEARNING_RATE = 0.005
    WEIGHT_DECAY = 5e-4
    EPOCHS = 200
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
