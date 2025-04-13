from torch.amp import autocast, GradScaler
from config import Config

DEVICE = Config.DEVICE


def train(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)

    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


def train_light(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    scaler = GradScaler()

    with autocast(device_type=DEVICE.type):
        out = model(data.x, data.edge_index)
        # Only calculate loss on training nodes
        loss = criterion(out[data.train_mask], data.y[data.train_mask])

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    return loss.item()
