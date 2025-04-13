from .evaluate import evaluate
import networkx as nx
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from node2vec import Node2Vec
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import pandas as pd


def comparison(data, model):
    results = {}
    test_acc, test_prec, test_rec, test_f1, test_auc = evaluate(
        model, data, data.test_mask
    )  # Pass the whole data object
    results["GAT"] = {
        "Accuracy": test_acc,
        "Precision": test_prec,
        "Recall": test_rec,
        "F1": test_f1,
        "AUC": test_auc,
    }
    data_cpu = data.cpu()  # Move data to CPU for processing

    def get_nx_graph(pyg_data):
        print("Converting PyG data to NetworkX graph...")
        if pyg_data.num_edges == 0:
            print("Warning: No edges in PyG data. Creating graph with nodes only.")
            G = nx.Graph()
            G.add_nodes_from(range(pyg_data.num_nodes))
            return G
        edge_list = pyg_data.edge_index.t().numpy()
        unique_edges = np.unique(np.sort(edge_list, axis=1), axis=0)
        G = nx.Graph()
        G.add_nodes_from(range(pyg_data.num_nodes))
        G.add_edges_from(unique_edges)
        print(
            f"NetworkX graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges."
        )
        return G

    nx_graph = get_nx_graph(data_cpu)

    # --- 3. Calculate PageRank ---
    print("Calculating PageRank...")
    if nx_graph.number_of_nodes() > 0:
        pagerank_scores = nx.pagerank(nx_graph, alpha=0.85, max_iter=1000)
        pr_feature = np.array(
            [pagerank_scores.get(i, 0) for i in range(data_cpu.num_nodes)]
        )
        print("PageRank calculation complete.")

        # Evaluate Direct PageRank Baseline (AUC)
        pr_scores_test = pr_feature[data_cpu.test_mask.numpy()]
        y_test_pr = data_cpu.y.numpy()[data_cpu.test_mask.numpy()]
        if len(pr_scores_test) > 0 and len(np.unique(y_test_pr)) > 1:
            try:
                pr_auc = roc_auc_score(y_test_pr, pr_scores_test)
            except ValueError:
                pr_auc = 0.5
        else:
            pr_auc = 0.5
        results["PageRank_Direct"] = {
            "Accuracy": "-",
            "Precision": "-",
            "Recall": "-",
            "F1": "-",
            "AUC": pr_auc,
        }
    else:
        print("Skipping PageRank calculation due to empty graph.")
        pr_feature = np.zeros((data_cpu.num_nodes, 1))  # Placeholder
        results["PageRank_Direct"] = {
            "Accuracy": "-",
            "Precision": "-",
            "Recall": "-",
            "F1": "-",
            "AUC": 0.0,
        }

    # --- 4. Calculate Common Neighbors Feature ---
    print("Calculating Common Neighbors feature...")
    cn_feature = np.zeros(data_cpu.num_nodes)
    if nx_graph.number_of_edges() > 0:
        for node in tqdm(nx_graph.nodes(), desc="Calculating CN Feature", leave=False):
            neighbors = list(nx_graph.neighbors(node))
            if not neighbors:
                continue
            total_cn_shared = sum(
                len(list(nx.common_neighbors(nx_graph, node, neighbor)))
                for neighbor in neighbors
            )
            cn_feature[node] = total_cn_shared / len(neighbors) if neighbors else 0
        print("Common Neighbors feature calculation complete.")
    else:
        print("Skipping CN calculation due to no edges.")

    # --- 5. Generate Node2Vec Embeddings ---
    # print("Generating Node2Vec embeddings...")
    # N2V_DIMENSIONS = 64  # Example dimension
    # n2v_embeddings = np.zeros(
    #     (data_cpu.num_nodes, N2V_DIMENSIONS)
    # )  # Default placeholder
    # if nx_graph.number_of_edges() > 0:
    #     try:
    #         N2V_WALK_LENGTH = 20
    #         N2V_NUM_WALKS = 10
    #         N2V_WORKERS = 4
    #         N2V_P = 1
    #         N2V_Q = 1
    #         node2vec_model = Node2Vec(
    #             nx_graph,
    #             dimensions=N2V_DIMENSIONS,
    #             walk_length=N2V_WALK_LENGTH,
    #             num_walks=N2V_NUM_WALKS,
    #             workers=N2V_WORKERS,
    #             p=N2V_P,
    #             q=N2V_Q,
    #             quiet=True,
    #         )
    #         n2v_trained_model = node2vec_model.fit(window=5, min_count=1, batch_words=4)
    #         # Ensure embeddings are fetched correctly, handling nodes not in vocab
    #         n2v_embeddings = np.array(
    #             [
    #                 (
    #                     n2v_trained_model.wv[node_id]
    #                     if node_id in n2v_trained_model.wv
    #                     else np.zeros(N2V_DIMENSIONS)
    #                 )
    #                 for node_id in range(data_cpu.num_nodes)
    #             ]
    #         )
    #         print("Node2Vec embedding generation complete.")
    #     except Exception as e:
    #         print(f"Error during Node2Vec execution: {e}. Using zero embeddings.")
    # else:
    #     print("Skipping Node2Vec due to no edges. Using zero embeddings.")

    # --- 6. Prepare Feature Sets & Train Baseline Classifiers ---
    print("Preparing feature sets and training baseline classifiers...")
    # Extract features and labels from the CPU version of the data object
    X_all = data_cpu.x.numpy()
    y_all = data_cpu.y.numpy()
    train_mask = data_cpu.train_mask.numpy()
    test_mask = data_cpu.test_mask.numpy()
    # val_mask = data_cpu.val_mask.numpy() # Needed for hyperparameter tuning (not implemented here)

    # Define feature sets (Ensure shapes are compatible)
    X_handcrafted = X_all
    X_hc_pr = np.concatenate([X_handcrafted, pr_feature[:, None]], axis=1)
    X_hc_cn = np.concatenate([X_handcrafted, cn_feature[:, None]], axis=1)
    # X_n2v = n2v_embeddings
    # X_n2v_hc = np.concatenate([X_n2v, X_handcrafted], axis=1)

    feature_sets = {
        "Handcrafted": X_handcrafted,
        "HC+PR": X_hc_pr,
        "HC+CN": X_hc_cn,
        # "Node2Vec": X_n2v,
        # "N2V+HC": X_n2v_hc,
    }
    baseline_models = {}

    # Train classifiers only if training data exists
    if train_mask.sum() > 0:
        for name, X in feature_sets.items():
            print(f"Training Logistic Regression for: {name}")
            # Check if features exist for this set after masking
            if X.shape[0] > 0 and X[train_mask].shape[0] > 0:
                pipeline = Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        (
                            "logreg",
                            LogisticRegression(
                                max_iter=1000,
                                class_weight="balanced",
                                C=1.0,
                                solver="liblinear",
                            ),
                        ),
                    ]
                )
                pipeline.fit(X[train_mask], y_all[train_mask])
                baseline_models[name] = pipeline
                # Note: For a rigorous comparison, hyperparameters (like C) should be tuned using the val_mask.
            else:
                print(
                    f"Skipping training for {name} due to empty training data after masking or feature preparation."
                )
                baseline_models[name] = None
    else:
        print("Warning: Training mask is empty! Skipping baseline classifier training.")

    print("Baseline classifier training complete.")

    # --- 7. Evaluate Baseline Classifiers on Test Set ---
    print("\nEvaluating baseline classifiers on Test Set...")
    for name, pipeline in baseline_models.items():
        if pipeline is None:  # Check if model exists
            print(f"Skipping evaluation for {name} (untrained).")
            results[f"LogReg_{name}"] = {
                "Accuracy": 0,
                "Precision": 0,
                "Recall": 0,
                "F1": 0,
                "AUC": 0,
            }
            continue

        # Prepare test features for the current model
        X_test_current = feature_sets[name][test_mask]
        y_test_current = y_all[test_mask]

        if (
            X_test_current.shape[0] > 0
        ):  # Check if there are samples in the test set for this feature set
            y_pred = pipeline.predict(X_test_current)
            try:
                y_prob = pipeline.predict_proba(X_test_current)[:, 1]
            except AttributeError:
                y_prob = y_pred  # Fallback if no predict_proba

            acc = accuracy_score(y_test_current, y_pred)
            prec = precision_score(y_test_current, y_pred, zero_division=0)
            rec = recall_score(y_test_current, y_pred, zero_division=0)
            f1 = f1_score(y_test_current, y_pred, zero_division=0)
            try:
                if len(np.unique(y_test_current)) < 2:
                    auc = 0.5
                else:
                    auc = roc_auc_score(y_test_current, y_prob)
            except ValueError:
                auc = 0.5
            results[f"LogReg_{name}"] = {
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1": f1,
                "AUC": auc,
            }
        else:
            print(
                f"Skipping evaluation for {name} due to empty test set after masking."
            )
            results[f"LogReg_{name}"] = {
                "Accuracy": 0,
                "Precision": 0,
                "Recall": 0,
                "F1": 0,
                "AUC": 0,
            }

    # --- 8. Display Final Results ---
    print("\n--- Final Comparison Results ---")
    # Check if results dictionary is populated before creating DataFrame
    if results:
        results_df = pd.DataFrame(results).T.sort_values(by="AUC", ascending=False)
        pd.options.display.float_format = "{:.4f}".format
        print(results_df)
        print("\n")
        print("\n")
    else:
        print("No evaluation results were generated.")
    print("---------------------------------")
