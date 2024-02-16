from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
import torch

def pool_graph_features(x, batch):
    # Assumption: this makes sense if the if the number of features is low (e.g. 2 or 3) to improve the representation of a graph
    mean_pool = global_mean_pool(x, batch)
    add_pool = global_add_pool(x, batch)
    max_pool = global_max_pool(x, batch)

    # Concatenating the pooled features
    pooled_features = torch.cat([mean_pool, add_pool, max_pool], dim=1)
    return pooled_features