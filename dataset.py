import os
import numpy as np
from torch_geometric.loader import ImbalancedSampler, DataLoader
from torch_geometric.data import InMemoryDataset, Data
import torch
from sklearn.model_selection import train_test_split

from data_structures import *

class PowerGrid(InMemoryDataset):
    # Base folder to download the files

    def __init__(self, data_save_dir, dataset: DatasetType, task_type: TaskType, transform=None, pre_transform=None, pre_filter=None):
        self.dataset = dataset.name.lower()
        self.task_type = task_type.name.lower()
        self.root = os.path.join(data_save_dir, self.dataset, self.task_type)
        
        
        self.pre_filter = pre_filter
        self.pre_transform = pre_transform

        super(PowerGrid, self).__init__(self.root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0]) 


    @property
    def processed_dir(self):    
        return self.root

    @property
    def processed_file_names(self):
        return 'data.pt'

def get_dataset(dataset: DatasetType, task_type: TaskType):
    data_save_dir = 'datasets'
    dataset = PowerGrid(data_save_dir, dataset, task_type)
    return dataset

def stratified_split(indices, labels):
    # First, split data into training and temp (val+test) sets
    train_indices, tmp_indices, _, tmp_labels = train_test_split(indices, labels, stratify=labels, test_size=0.2, random_state=42)

    # Then, split temp into validation and test sets
    val_indices, test_indices, _, _ = train_test_split(tmp_indices, tmp_labels, stratify=tmp_labels, test_size=0.5, random_state=42)
    return train_indices, val_indices, test_indices

def get_test_dataloader(dataset_type, task_type, batch_size):
    dataset = get_dataset(
            dataset=dataset_type,
            task_type=task_type,
        )

    dataset._data.x = dataset._data.x.float()
    
    if task_type == TaskType.MULTICLASS:
        dataset._data.y = dataset._data.y.squeeze().long()

    num_node_features = dataset.num_node_features
    num_edge_features = dataset.num_edge_features  
    num_classes = dataset.num_classes
    labels = dataset.data.y.detach().cpu().numpy()
    unique_labels, counts = np.unique(labels, return_counts=True)

    dataset_characteristics = {
        "num_node_features": num_node_features,
        "num_edge_features": num_edge_features,
        "num_classes": num_classes,
        "unique_labels": list(unique_labels),
        "counts": list(counts),
        "dataset": dataset_type.name.lower()
    }

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader, dataset_characteristics

# Use exising code from the dataset
def get_dataloaders(datasets: DatasetType, task_type: TaskType, batch_size: int):  
       
    dataset = get_dataset(
            dataset=datasets,
            task_type=task_type,
        )

    dataset._data.x = dataset._data.x.float()
    
    if task_type == TaskType.MULTICLASS:
        dataset._data.y = dataset._data.y.squeeze().long()

    num_node_features = dataset.num_node_features
    num_edge_features = dataset.num_edge_features  
    num_classes = dataset.num_classes
    labels = dataset.data.y.detach().cpu().numpy()
    unique_labels, counts = np.unique(labels, return_counts=True)

    dataset_characteristics = {
        "num_node_features": num_node_features,
        "num_edge_features": num_edge_features,
        "num_classes": num_classes,
        "unique_labels": list(unique_labels),
        "counts": list(counts),
        "dataset": datasets.name.lower()
    }

    loader = dict()

    indices = np.arange(len(labels))
    train_indices, val_indices, test_indices = stratified_split(indices, labels)
    train_subset = dataset[train_indices]
    val_subset = dataset[val_indices]
    test_subset = dataset[test_indices]
    loader['train'] = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    loader['eval'] = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    loader['test'] = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
   
    return loader, dataset_characteristics