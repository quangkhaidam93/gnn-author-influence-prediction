import json
import numpy as np
import torch
from torch_geometric.data import (
    Data,
)


def load_data(data_path: str, influence_threshold: float):
    num_authors = 2153278
    author_ids = list(range(num_authors))
    author_features = []
    author_citations_t = {}
    author_citations_t_plus_w = {}
    edges = set()

    with open(data_path, "r") as file:
        for line in file:

            author = json.loads(line)

            author_id = author["id"]

            features = np.array(
                [
                    author["papers"],
                    author["citations"],
                    author["avg_pqi"],
                    len(author["collaborators"]),
                    author["avg_year_published"],
                    author["activity_score"],
                    author["recent_citations"],
                ]
            )

            author_features.append(features)

            author_citations_t[author_id] = author["citations"]
            author_citations_t_plus_w[author_id] = author["citations_t_plus_w"]

            for collaborator_id in author["collaborators"]:
                if author_id < collaborator_id:
                    edges.add((author_id, collaborator_id))
                else:
                    edges.add((collaborator_id, author_id))

        edge_list = list(edges)
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        x = torch.tensor(np.array(author_features), dtype=torch.float)

        y = []

        for author_id in author_ids:
            increase = (
                author_citations_t_plus_w[author_id] - author_citations_t[author_id]
            )
            target = 1 if increase > influence_threshold else 0
            y.append(target)

        y = torch.tensor(y, dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, y=y)
        data.num_classes = 2  # Binary classification (increase > threshold or not)

        num_nodes = data.num_nodes
        indices = np.arange(num_nodes)
        np.random.shuffle(indices)

        train_ratio, val_ratio = 0.7, 0.15
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

        return data
