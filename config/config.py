from datetime import datetime

import torch


class Config:
    CURRENT_YEAR = datetime.now().year
    PREDICTION_WINDOW = 3  # Predict increase over W years
    INFLUENCE_THRESHOLD = (
        10  # Absolute increase in citations to be considered 'influential'
    )
    HIDDEN_CHANNELS = 64  # GNN hidden layer size
    NUM_HEADS = 4  # Number of attention heads in GAT
    LEARNING_RATE = 0.005
    WEIGHT_DECAY = 5e-4
    EPOCHS = 200
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
