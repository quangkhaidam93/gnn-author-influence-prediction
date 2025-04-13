from datetime import datetime

import torch


class Config:
    CURRENT_YEAR = datetime.now().year
    PREDICTION_WINDOW = 3
    INFLUENCE_THRESHOLD = 10
    HIDDEN_CHANNELS = 64
    NUM_HEADS = 4
    LEARNING_RATE = 0.005
    WEIGHT_DECAY = 5e-4
    EPOCHS = 200
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
