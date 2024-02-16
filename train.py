import os
import copy
from loguru import logger
from tqdm import tqdm
import json
import pandas as pd

from dataset import *
from trainer import ModelTrainer
from model import GNNModel 

def run_experiment(parameters, number_of_iterations: int = 1, debug: bool = False, return_results: bool = True):
    experiment_results = list()

    project_dir = '/var/repos/michael-repos/power_grids/powergrids_cascading_failures_detection'

    experiments_dir = os.path.join(project_dir, 'experiments', 'gnn')
    os.makedirs(experiments_dir, exist_ok=True)
    experiment_number = sum(os.path.isdir(os.path.join(experiments_dir, d)) for d in os.listdir(experiments_dir))
    experiment_dir = os.path.join(experiments_dir, f'experiment_{experiment_number}')
    os.makedirs(experiment_dir, exist_ok=True)

    logger.add(os.path.join(experiment_dir, 'experiment.log'))
    logger.info(f"Running experiments with {len(list(copy.copy(parameters)))} different parameter combinations and {number_of_iterations} iterations.")

    
    
    for _ in range(number_of_iterations):
        for params in tqdm(list(copy.copy(parameters))):
           
            hidden_dim, num_conv_layer, num_linear_layers, conv_type, pooling, task_type, dataset, batch_size = params

            experiment_result = dict()

            experiment_result['setup'] = {
                "hidden_dim": hidden_dim,
                "num_conv_layer": num_conv_layer,
                "num_linear_layers": num_linear_layers,
                "conv_type": conv_type.name.lower(),
                "pooling": pooling.name.lower(),
                "batch_size": batch_size,
                "task_type": task_type.name.lower(),
                "dataset": dataset.name.lower(),
            }


            logger.info(f"Training model with {experiment_result['setup']}")
            loader, dataset_characteristics = get_dataloaders(dataset, task_type, batch_size)
            logger.info(f"Dataset characteristics: {dataset_characteristics}")
            
            # Create a directory to save model and results
            tmp_dir = f"models/gnn/{dataset.name}_{task_type.name}_{conv_type.name}_{hidden_dim}_{num_conv_layer}_{num_linear_layers}_{pooling.name}_{batch_size}"
            output_dir = os.path.join(project_dir, tmp_dir)
            os.makedirs(output_dir, exist_ok=True)

            # Initialize model, optimizer, etc.
            model = GNNModel(dataset_characteristics['num_node_features'], dataset_characteristics['num_edge_features'], dataset_characteristics['num_classes'], hidden_dim, num_conv_layer, num_linear_layers, conv_type, pooling, task_type)
            
            trainer = ModelTrainer(model, loader, dataset_characteristics, task_type, output_dir)
            result = trainer.train_and_evaluate_model(debug=debug)

            experiment_result['results'] = result
            experiment_results.append(experiment_result)
          
   

    # Write the dictionary to a JSON file
    with open(os.path.join(experiment_dir, 'results.json'), 'w') as file:
        json.dump(experiment_results, file, indent=4)

    flattened_data = []
    for item in experiment_results:
        if item['setup']['dataset'] == DatasetType.ALL.name.lower():
            for key in ['all', 'ieee39', 'ieee118']:
                flat_item = {}
                item['setup']['test_dataset'] = key
                flat_item.update(item['setup'])
                flat_item.update(item['results'][key])
                flattened_data.append(flat_item)
        else:
            flat_item = {}
            item['setup']['test_dataset'] = item['setup']['dataset']
            flat_item.update(item['setup'])
            flat_item.update(item['results'][item['setup']['dataset']])
            flattened_data.append(flat_item)

    # Creating DataFrame
    df = pd.DataFrame(flattened_data)

    # Rounding float values to 2 decimal places
    df = df.round(3)

    # Expand explainability column:
    if 'explainability' in df.columns:
        df = df.join(pd.json_normalize(df['explainability']))
        df.drop(columns=['explainability'], inplace=True)

    df.to_csv(os.path.join(experiment_dir, 'results.csv'), index=True)
    
    if return_results:  
        return experiment_results


def main():
    from itertools import product

    # Define Parameter Grid
    number_of_iterations = 1
    hidden_dims = [32]
    num_conv_layers = [3]
    num_linear_layers = [2]
    conv_types = [ConvType.TRANSFORMER]
    task_types = [TaskType.MULTICLASS, TaskType.BINARY]
    datasets = [DatasetType.IEEE118, DatasetType.IEEE39]
    batch_sizes = [128]
    poolings = [PoolingType.MEAN]


    # Create a grid of all parameter combinations
    param_grid = product(hidden_dims, num_conv_layers, num_linear_layers, conv_types, poolings, task_types, datasets, batch_sizes)

    run_experiment(param_grid, number_of_iterations, debug=False)



if __name__ == "__main__":
    main()