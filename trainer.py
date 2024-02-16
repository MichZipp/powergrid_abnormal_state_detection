import copy
import os
import json
import numpy as np
import torch
from loguru import logger
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support

from data_structures import *

class ModelTrainer():
    def __init__(self, model, dataset_characteristics, task_type):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = model.to(self.device)
        self.dataset_characteristics = dataset_characteristics
        self.task_type = task_type


        self.model_dir = f"models/{dataset_characteristics['dataset'].upper()}_{task_type.name}_{conv_layer_reverse_mapping[model.conv_layers[0].__class__].name}_{model.conv_layers[0].out_channels}_{len(model.conv_layers)}_{len(model.linear_layers)}_{pooling_reverse_mapping[model.pooling].name}"

        self.num_epochs = 100 
        self.best_model_wts = copy.deepcopy(model.state_dict())
        self.best_loss = float('inf')
        self.epochs_no_improve = 0
        

        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=3, verbose=True)

        class_sample_counts = self.dataset_characteristics['counts']

        # Calculate weights
        weights = 1. / torch.tensor(class_sample_counts, dtype=torch.float)            
        # Normalize weights
        weights = weights / weights.sum()


        if task_type is TaskType.BINARY:
            # self.criterion = torch.nn.BCELoss(weight=weights)
            # self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=weights[1] / weights[0]).to(self.device)
            pos_weight = pos_weight=weights[1] / weights[0]
            self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(self.device)
        elif task_type is TaskType.MULTICLASS:
            self.criterion = torch.nn.CrossEntropyLoss(weight=weights).to(self.device)
    
    def set_data_loaders(self, loader):
        self.train_loader = loader['train']
        self.eval_loader = loader['eval']
        self.test_loader = loader['test']

        if self.dataset_characteristics['dataset'] == DatasetType.ALL.name.lower():
            self.eval_loader_ieee39 = loader['eval_ieee39']
            self.eval_loader_ieee118 = loader['eval_ieee118']
            self.test_loader_ieee39 = loader['test_ieee39']
            self.test_loader_ieee118 = loader['test_ieee118']

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0
        for data in self.train_loader:
            data = data.to(self.device)
            if self.augmentation:
                data = self.perturb_edge_features_consistently(data)
            self.optimizer.zero_grad()
            out = self.model(data.x, data.edge_index, data.edge_attr, data.batch)
            if self.task_type in [TaskType.BINARY, TaskType.BINARY_CF, TaskType.BINARY_DNS]:
                target = data.y.view(-1, 1).float()
            else:
                target = data.y
            loss = self.criterion(out, target)
            loss.backward()
            self.optimizer.step()
            total_loss += data.num_graphs * loss.item()
        return total_loss / len(self.train_loader.dataset)

    def validate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for data in self.eval_loader:
                data = data.to(self.device)
                out = self.model(data.x, data.edge_index, data.edge_attr, data.batch)
                if self.task_type in [TaskType.BINARY, TaskType.BINARY_CF, TaskType.BINARY_DNS]:
                    target = data.y.view(-1, 1).float()
                else:
                    target = data.y
                loss = self.criterion(out, target)
                total_loss += data.num_graphs * loss.item()
        return total_loss / len(self.eval_loader.dataset)

    def calculate_confusion_matrix(self, all_labels, all_preds):
        TP = FP = TN = FN = 0

        for actual, predicted in zip(all_labels, all_preds):
            if actual == 1 and predicted == 1:
                TP += 1
            elif actual == 0 and predicted == 1:
                FP += 1
            elif actual == 0 and predicted == 0:
                TN += 1
            elif actual == 1 and predicted == 0:
                FN += 1

        return TP, TN, FP, FN
    
    def get_reachable_edges(self, edge_index, edge_of_interest_idx, hops = 1):
        # Get the target node of the edge of interest
        target_node = edge_index[1, edge_of_interest_idx]
        source_node = edge_index[0, edge_of_interest_idx]
        
        # Find edges where this node is the source
        reachable_edges_mask = (edge_index[0] == target_node) + (edge_index[1] == source_node)
        # reachable_edges = edge_index[:, reachable_edges_mask]
        indices = np.where(reachable_edges_mask == 1)[0]

        # Recursively find reachable edges
        if hops > 1:
            for i in indices:
                indices = np.concatenate((indices, self.get_reachable_edges(edge_index, i, hops - 1)))
        return indices
            
    
    def evaluate(self, loader):
        self.model.eval()
        all_preds = []
        all_labels = []

        # with torch.no_grad():
        for data in loader:
            data = data.to(self.device)
            data.x.requires_grad_(True)
            data.edge_attr.requires_grad_(True)
            out = self.model(data.x, data.edge_index, data.edge_attr, data.batch)
            if self.task_type is TaskType.BINARY:
                probabilities = torch.sigmoid(out)
                # Apply threshold to get binary predictions
                preds = (probabilities >= 0.5).int()
            else:
                preds = out.argmax(dim=1)
    
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        balanced_accuracy = balanced_accuracy_score(all_labels, all_preds, adjusted=False)
        if self.task_type is TaskType.BINARY:
            recall, precision, specificity, sensitivity, f1 = 0, 0, 0, 0, 0
            # TP, TN, FP, FN = self.calculate_confusion_matrix(all_labels, all_preds)
            TP, TN, FP, FN = self.calculate_confusion_matrix(all_labels, all_preds)
            logger.info(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
            precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0.0)
            if TP + FN != 0:
                recall = TP / (TP + FN)
            if TP + FP != 0:
                precision = TP / (TP + FP)
            if TN + FP != 0:
                specificity = TN / (TN + FP)
            sensitivity = recall 
            balanced_accuracy = (sensitivity + specificity) / 2
            result = {
                'accuracy': accuracy,
                'balanced_accuracy': balanced_accuracy,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'TP': TP,
                'TN': TN,
                'FP': FP,
                'FN': FN
            }
        else:
            precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0.0)
            sensitivity = recall
            specificity = 0
            result = {
                'accuracy': accuracy,
                'balanced_accuracy': balanced_accuracy,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }

        return result    


    def train_and_evaluate_model(self, debug: bool = False):
        for epoch in range(self.num_epochs):
            train_loss = self.train_one_epoch()            
            
            # Validate the model
            val_loss = self.validate()
            
            train_result = self.evaluate(self. train_loader)

            val_result = self.evaluate(self.eval_loader)

            # Print training and validation results
            logger.info(f'Epoch {epoch+1}/{self.num_epochs}')

            logger.info(f'TRAIN loss {train_loss:.4f}, Accuracy: {train_result["accuracy"]:.4f}, Balanced Accuracy: {train_result["balanced_accuracy"]:.4f}, Precision: {train_result["precision"]:.4f}, Recall: {train_result["recall"]:.4f}, F1: {train_result["f1"]:.4f}')

            logger.info(f'VAL loss {val_loss:.4f}, Accuracy: {val_result["accuracy"]:.4f}, Balanced Accuracy: {val_result["balanced_accuracy"]:.4f}, Precision: {val_result["precision"]:.4f}, Recall: {val_result["recall"]:.4f}, F1: {val_result["f1"]:.4f}')

            # Learning rate scheduling based on validation loss
            self.scheduler.step(val_loss)

            # Check if this is the best model (based on validation loss)
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.best_model_wts = copy.deepcopy(self.model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve == self.patience:
                    logger.info('Early stopping')
                    break
            
            if debug:
                break

        # Load the best model weights
        self.model.load_state_dict(self.best_model_wts)
        
        # Save the model
        torch.save(self.model.state_dict(), os.path.join(self.model_dir, 'model.pth'))

        # Evaluate on the test set
        
        result = self.evaluate(self.test_loader)

        logger.info(f'TEST Accuracy: {result["accuracy"]:.4f}, Balanced Accuracy: {result["balanced_accuracy"]:.4f}, Precision: {result["precision"]:.4f}, Recall: {result["recall"]:.4f}, F1: {result["f1"]:.4f}')

        result = {
            self.dataset_characteristics['dataset']: result
        } 

        # Save the results
        with open(os.path.join(self.model_dir, 'results.json'), 'w') as file:
            json.dump(result, file, indent=4)       

        return result
    
    def load_model_state(self):
        self.model.load_state_dict(torch.load(os.path.join(self.model_dir, 'model.pth')))

    def evaluate_model(self, loader):
        self.load_model_state()
        result = self.evaluate(loader)
        logger.info(f'TEST results: {result}')
        return result