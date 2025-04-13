import torch
from torch.amp import autocast, GradScaler
from torch_geometric.nn import GATConv
from torch.nn import BatchNorm1d
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from data import load_data

INFLUENCE_THRESHOLD = (
    10  # Absolute increase in citations to be considered 'influential'
)
HIDDEN_CHANNELS = 30  # GNN hidden layer size
NUM_HEADS = 2  # Number of attention heads in GAT
LEARNING_RATE = 0.005
WEIGHT_DECAY = 5e-4
EPOCHS = 100000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- 2. Define GAT Model ---
class AuthorGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads):
        super().__init__()
        # Increase hidden_channels fed into the second GAT layer because concatenation
        # multiplies the channel dimension by the number of heads.
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.6)

        # Optional: BatchNorm can stabilize training
        self.bn1 = BatchNorm1d(hidden_channels * heads)

        # Use heads=1 or average the heads for the final layer output
        self.conv2 = GATConv(
            hidden_channels * heads, out_channels, heads=1, concat=False, dropout=0.6
        )

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        # No activation before final output for classification loss
        x = self.conv2(x, edge_index)
        # Return logits
        return x


# --- 3. Training Function ---
def train(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    # Only calculate loss on training nodes
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


# --- 4. Evaluation Function ---
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
    try:
        auc = roc_auc_score(label.cpu(), prob.cpu())
    except ValueError:  # Handle cases where only one class is present in the mask
        auc = 0.5  # Or another appropriate default

    return acc, prec, rec, f1, auc


# --- 5. Main Execution ---
if __name__ == "__main__":
    # --- Load Data ---
    # Replace 'path/to/your/processed/data' if needed by your loader
    data = load_data("data/authors.jsonl", INFLUENCE_THRESHOLD)
    data = data.to(DEVICE)

    # --- Initialize Model ---
    model = AuthorGAT(
        in_channels=data.num_node_features,
        hidden_channels=HIDDEN_CHANNELS,
        out_channels=data.num_classes,
        heads=NUM_HEADS,
    ).to(DEVICE)

    # --- Optimizer and Loss ---
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    criterion = (
        torch.nn.CrossEntropyLoss()
    )  # Suitable for multi-class (here binary) classification

    # --- Training Loop ---
    print("\n--- Starting Training ---")
    best_val_auc = 0
    best_epoch = 0
    for epoch in range(1, EPOCHS + 1):
        loss = train(model, data, optimizer, criterion)
        # Evaluate on validation set
        val_acc, val_prec, val_rec, val_f1, val_auc = evaluate(
            model, data, data.val_mask
        )

        if (
            val_auc > best_val_auc
        ):  # Use AUC or F1-score for selecting best model, especially if data is imbalanced
            best_val_auc = val_auc
            best_epoch = epoch
            # Save the best model checkpoint
            torch.save(model.state_dict(), "best_gat_model.pth")
            print(
                f"[Epoch {epoch:03d}] Loss: {loss:.4f} | Val Acc: {val_acc:.4f} | Val AUC: {val_auc:.4f} ** (New Best)**"
            )
        else:
            if epoch % 10 == 0:  # Print periodically
                print(
                    f"[Epoch {epoch:03d}] Loss: {loss:.4f} | Val Acc: {val_acc:.4f} | Val AUC: {val_auc:.4f}"
                )

    print(
        f"--- Training Finished --- Best validation AUC {best_val_auc:.4f} at epoch {best_epoch} ---"
    )

    # --- 6. Inference and Final Evaluation ---
    print("\n--- Evaluating on Test Set using Best Model ---")
    # Load the best model weights
    model.load_state_dict(torch.load("best_gat_model.pth"))

    test_acc, test_prec, test_rec, test_f1, test_auc = evaluate(
        model, data, data.test_mask
    )

    print(f"Test Accuracy:  {test_acc:.4f}")
    print(f"Test Precision: {test_prec:.4f}")
    print(f"Test Recall:    {test_rec:.4f}")
    print(f"Test F1-Score:  {test_f1:.4f}")
    print(f"Test AUC:       {test_auc:.4f}")

    # --- Example Inference for a specific author (if needed) ---
    def predict_author_influence(author_node_id, model, data):
        model.eval()
        with torch.no_grad():
            out = model(data.x.to(DEVICE), data.edge_index.to(DEVICE))
            author_logits = out[author_node_id]
            author_probs = F.softmax(author_logits, dim=0)
            predicted_class = author_probs.argmax().item()
            probability_increase = author_probs[
                1
            ].item()  # Probability of class 1 (increase > threshold)

            print(f"\nInference for Author ID: {author_node_id}")
            print(
                f"  Predicted Class: {'Increase Expected' if predicted_class == 1 else 'Increase Not Expected'}"
            )
            print(f"  Probability of Significant Increase: {probability_increase:.4f}")
        return predicted_class, probability_increase

    # Example: Predict for the first author in the test set
    test_author_indices = data.test_mask.nonzero(as_tuple=True)[0]
    if len(test_author_indices) > 0:
        predict_author_influence(test_author_indices[0].item(), model, data)
        if len(test_author_indices) > 1:
            predict_author_influence(test_author_indices[1].item(), model, data)
    else:
        print("\nNo authors in the simulated test set to run inference on.")
