import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
import torch.nn.functional as F


@torch.no_grad()  # Disable gradient calculations for evaluation
def evaluate(model, data, mask):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out[mask].argmax(dim=1)  # Get predicted class (0 or 1)
    label = data.y[mask]
    prob = F.softmax(out[mask], dim=1)[:, 1]  # Probability of class 1

    acc = accuracy_score(label.cpu(), pred.cpu())
    prec = precision_score(label.cpu(), pred.cpu(), zero_division=0)
    rec = recall_score(label.cpu(), pred.cpu(), zero_division=0)
    f1 = f1_score(label.cpu(), pred.cpu(), zero_division=0)
    avg_prec = average_precision_score(label.cpu(), prob.cpu())
    try:
        auc = roc_auc_score(label.cpu(), prob.cpu())
    except ValueError:  # Handle cases where only one class is present in the mask
        auc = 0.5  # Or another appropriate default

    return acc, prec, rec, f1, auc, avg_prec
