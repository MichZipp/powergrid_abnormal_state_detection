import torch
from torch.nn import Linear, ModuleList
import torch.nn.functional as F
from data_structures import *
from dataset import *



class GNNModel(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, num_classes, hidden_dim, num_conv_layers, num_linear_layers, conv_type, pooling, task_type):
        super(GNNModel, self).__init__()

        self.conv_layers = ModuleList()
        self.linear_layers = ModuleList()
        self.task_type = task_type


        self.pooling = pooling_mapping[pooling]

        conv_layer = conv_layer_mapping[conv_type]

        # First layer
        self.conv_layers.append(conv_layer(num_node_features, hidden_dim, edge_dim=num_edge_features))

        # Additional layers
        for _ in range(1, num_conv_layers):
            self.conv_layers.append(conv_layer(hidden_dim, hidden_dim, edge_dim=num_edge_features))

        # Output layer
        for index in range(1, num_linear_layers+1):
            if index == num_linear_layers:
                
                if pooling == PoolingType.ALL:
                    self.linear_layers.append(Linear(3* hidden_dim, num_classes))
                else:
                    if self.task_type is TaskType.BINARY:
                        self.linear_layers.append(Linear(hidden_dim, 1))
                    else:
                        self.linear_layers.append(Linear(hidden_dim, num_classes))
            else:
                if pooling == PoolingType.ALL:
                    self.linear_layers.append(Linear(3 * hidden_dim, 3 * hidden_dim))
                else:
                    self.linear_layers.append(Linear(hidden_dim, hidden_dim))


        self.activation = torch.nn.LeakyReLU()
        
    

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        # Applying convolutional layers
        for layer in self.conv_layers:
            x = layer(x, edge_index, edge_attr=edge_attr if edge_attr is not None else None)
            x = self.activation(x)
            
        x = self.pooling(x, batch)

        # Output layer
        for i, layer in enumerate(self.linear_layers):
            x = layer(x)
            # Apply ReLU activation to all but the last layer
            if i < len(self.linear_layers) - 1:
                x = self.activation(x)

        if self.task_type is TaskType.BINARY:
            return x
        else: 
            return F.log_softmax(x, dim=1)
        





