from torch_geometric.data import (
    Data,
    DataLoader,
)
import numpy as np
import torch


def load_and_preprocess_data(
    data_path: str, current_year, prediction_window, influence_threshold
):
    """
    Simulates loading and preprocessing data to create an author graph.
    Replace this with your actual data loading logic for DBLP etc.
    Args:
        data_path: Path to the DBLP XML file
        current_year: The cutoff year for training data (T)
        prediction_window: Number of years to predict ahead (W)
        influence_threshold: The threshold for citation increase to be considered influential

    Returns:
        A PyTorch Geometric Data object with the author graph
    """
    num_authors = 1000
    num_papers_per_author_avg = 10
    max_collaborators = 50
    feature_dim = 10  # Example: avg topic vector dim

    author_ids = list(range(num_authors))
    author_features = []
    author_citations_t = {}
    author_citations_t_plus_w = {}
    edges = set()  # Using a set to avoid duplicate edges

    print(f"Simulating data up to year T = {current_year}")

    for author_id in author_ids:
        # Simulate features at time T (current_year)
        papers_t = np.random.randint(1, num_papers_per_author_avg * 2)
        citations_t = np.random.randint(0, 500)  # Total citations up to year T
        avg_pqi_t = np.random.rand() * 10
        avg_collabs_t = np.random.randint(1, 10)
        topic_vec_t = np.random.rand(feature_dim)
        avg_pub_year_t = current_year - np.random.rand() * 10
        recent_activity_t = np.random.rand()  # Higher if published recently before T

        features = np.concatenate(
            [
                np.array(
                    [
                        papers_t,
                        citations_t,
                        avg_pqi_t,
                        avg_collabs_t,
                        avg_pub_year_t,
                        recent_activity_t,
                    ]
                ),
                topic_vec_t,
            ]
        )
        author_features.append(features)
        author_citations_t[author_id] = citations_t

        # Simulate citations at time T+W
        # Increase should depend somewhat on citations_t and maybe features
        increase_factor = 1.0 + np.random.rand() * 0.5  # Random increase up to 50%
        noise = np.random.randint(-5, 20)
        citations_t_plus_w = max(0, int(citations_t * increase_factor) + noise)
        author_citations_t_plus_w[author_id] = citations_t_plus_w

        # Simulate co-authorship edges
        num_collabs = np.random.randint(1, max_collaborators)
        collaborators = np.random.choice(
            [a for a in author_ids if a != author_id], num_collabs, replace=False
        )
        for collab_id in collaborators:
            # Add edges in both directions for undirected graph
            if author_id < collab_id:
                edges.add((author_id, collab_id))
            else:
                edges.add((collab_id, author_id))

    edge_list = list(edges)
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    x = torch.tensor(np.array(author_features), dtype=torch.float)

    # --- Calculate Target Variable ---
    y = []
    print(
        f"Calculating target based on influence increase threshold = {influence_threshold}"
    )
    for author_id in author_ids:
        increase = author_citations_t_plus_w[author_id] - author_citations_t[author_id]
        target = 1 if increase > influence_threshold else 0
        y.append(target)
    y = torch.tensor(y, dtype=torch.long)

    # --- Create PyG Data Object ---
    data = Data(x=x, edge_index=edge_index, y=y)
    data.num_classes = 2  # Binary classification (increase > threshold or not)

    # --- Create Train/Val/Test Masks (Temporal Split Simulation) ---
    # Ideally, split authors based on when they were 'active' or use
    # different time periods (T1 for train, T2 for val, T3 for test)
    # Here, we simulate a random split for simplicity, but this is NOT
    # temporally correct for a real scenario.
    print("Simulating Train/Val/Test split (NOTE: Real split should be temporal!)")
    num_nodes = data.num_nodes
    indices = np.arange(num_nodes)
    np.random.shuffle(indices)

    train_ratio, val_ratio = 0.7, 0.15  # Test ratio is 1 - train - val
    train_end = int(train_ratio * num_nodes)
    val_end = int((train_ratio + val_ratio) * num_nodes)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[indices[:train_end]] = True
    val_mask[indices[train_end:val_end]] = True
    test_mask[indices[val_end:]] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    print(f"Data created: {data}")
    print(f"Number of authors (nodes): {data.num_nodes}")
    print(
        f"Number of co-authorships (edges): {data.num_edges // 2}"
    )  # Divided by 2 for undirected
    print(f"Number of features per author: {data.num_node_features}")
    print(f"Number of training nodes: {data.train_mask.sum().item()}")
    print(f"Number of validation nodes: {data.val_mask.sum().item()}")
    print(f"Number of test nodes: {data.test_mask.sum().item()}")
    print(f"Target distribution (0/1): {np.bincount(data.y.numpy())}")

    return data
