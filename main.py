import torch
import torch.nn.functional as F
from data import load_data
from services import AuthorGAT, train, evaluate
from config import Config

INFLUENCE_THRESHOLD = Config.INFLUENCE_THRESHOLD
HIDDEN_CHANNELS = Config.HIDDEN_CHANNELS
NUM_HEADS = Config.NUM_HEADS
LEARNING_RATE = Config.LEARNING_RATE
WEIGHT_DECAY = Config.WEIGHT_DECAY
EPOCHS = Config.EPOCHS
DEVICE = Config.DEVICE

processed_authors_data = "data/authors.jsonl"


def train_pipeline():
    data = load_data(processed_authors_data, INFLUENCE_THRESHOLD)
    data = data.to(DEVICE)

    model = AuthorGAT(
        in_channels=data.num_node_features,
        hidden_channels=HIDDEN_CHANNELS,
        out_channels=data.num_classes,
        heads=NUM_HEADS,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    criterion = torch.nn.CrossEntropyLoss()

    print("\n--- Starting Training ---")
    best_val_auc = 0
    best_epoch = 0
    for epoch in range(1, EPOCHS + 1):
        loss = train(model, data, optimizer, criterion)

        val_acc, val_prec, val_rec, val_f1, val_auc = evaluate(
            model, data, data.val_mask
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch

            torch.save(model.state_dict(), "best_gat_model.pth")
            print(
                f"[Epoch {epoch:03d}] Loss: {loss:.4f} | Val Acc: {val_acc:.4f} | Val AUC: {val_auc:.4f} ** (New Best)**"
            )
        else:
            if epoch % 10 == 0:
                print(
                    f"[Epoch {epoch:03d}] Loss: {loss:.4f} | Val Acc: {val_acc:.4f} | Val AUC: {val_auc:.4f}"
                )

    print(
        f"--- Training Finished --- Best validation AUC {best_val_auc:.4f} at epoch {best_epoch} ---"
    )

    print("\n--- Evaluating on Test Set using Best Model ---")


def evaluate_pipeline():
    data = load_data(processed_authors_data, INFLUENCE_THRESHOLD)
    data = data.to(DEVICE)

    model = AuthorGAT(
        in_channels=data.num_node_features,
        hidden_channels=HIDDEN_CHANNELS,
        out_channels=data.num_classes,
        heads=NUM_HEADS,
    ).to(DEVICE)

    model.load_state_dict(torch.load("best_gat_model.pth"))

    test_acc, test_prec, test_rec, test_f1, test_auc = evaluate(
        model, data, data.test_mask
    )

    print(f"Test Accuracy:  {test_acc:.4f}")
    print(f"Test Precision: {test_prec:.4f}")
    print(f"Test Recall:    {test_rec:.4f}")
    print(f"Test F1-Score:  {test_f1:.4f}")
    print(f"Test AUC:       {test_auc:.4f}")


def inference_pipeline():
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

    data = load_data(processed_authors_data, INFLUENCE_THRESHOLD)
    data = data.to(DEVICE)

    model = AuthorGAT(
        in_channels=data.num_node_features,
        hidden_channels=HIDDEN_CHANNELS,
        out_channels=data.num_classes,
        heads=NUM_HEADS,
    ).to(DEVICE)

    model.load_state_dict(torch.load("best_gat_model.pth"))

    test_author_indices = data.test_mask.nonzero(as_tuple=True)[0]
    if len(test_author_indices) > 0:
        predict_author_influence(test_author_indices[0].item(), model, data)
        if len(test_author_indices) > 1:
            predict_author_influence(test_author_indices[1].item(), model, data)
    else:
        print("\nNo authors in the simulated test set to run inference on.")


if __name__ == "__main__":
    # train_pipeline()
    evaluate_pipeline()
    # inference_pipeline()
