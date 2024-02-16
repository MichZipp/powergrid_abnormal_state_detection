from enum import Enum, auto
from torch_geometric.nn import GATConv, GATv2Conv, TransformerConv, global_mean_pool, global_max_pool, global_add_pool
from pooling import pool_graph_features

class ConvType(Enum):
    GAT = auto()
    GATv2 = auto()
    TRANSFORMER = auto()

class TaskType(Enum):
    MULTICLASS = auto()
    BINARY = auto()

class DatasetType(Enum):
    IEEE39 = auto()
    IEEE118 = auto()

class PoolingType(Enum):
    MEAN = auto()
    MAX = auto()
    SUM = auto()
    ALL = auto()

#  Convolution layer mapping
conv_layer_mapping = {
    ConvType.GAT: GATConv,
    ConvType.GATv2: GATv2Conv,
    ConvType.TRANSFORMER: TransformerConv
}

conv_layer_reverse_mapping = {
    GATConv: ConvType.GAT,
    GATv2Conv: ConvType.GATv2,
    TransformerConv: ConvType.TRANSFORMER
}

# Pooling layer mapping
pooling_mapping = {
    PoolingType.MEAN: global_mean_pool,
    PoolingType.SUM: global_add_pool,
    PoolingType.MAX: global_max_pool,
    PoolingType.ALL: pool_graph_features
}

pooling_reverse_mapping = {
    global_mean_pool: PoolingType.MEAN,
    global_add_pool: PoolingType.SUM,
    global_max_pool: PoolingType.MAX,
    pool_graph_features: PoolingType.ALL
}