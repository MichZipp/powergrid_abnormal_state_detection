import os
import copy
from loguru import logger
from tqdm import tqdm
import json
import pandas as pd
from itertools import product

from dataset import *
from trainer import ModelTrainer
from model import GNNModel 

# Get the directory of the current file
current_file_directory = os.path.dirname(os.path.abspath(__file__))


def run_experiment(parameters, number_of_iterations: int = 1, debug: bool = False, return_results: bool = True):
    experiment_results = list()

    experiments_dir = os.path.join(current_file_directory, 'experiments')
    os.makedirs(experiments_dir, exist_ok=True)
    experiment_number = sum(os.path.isdir(os.path.join(experiments_dir, d)) for d in os.listdir(experiments_dir))
    experiment_dir = os.path.join(experiments_dir, f'experiment_{experiment_number}')
    os.makedirs(experiment_dir, exist_ok=True)

    logger.add(os.path.join(current_file_directory, 'experiment.log'))
    logger.info(f"Running experiments with {len(list(copy.copy(parameters)))} different parameter combinations and {number_of_iterations} iterations.")
    
    for _ in range(number_of_iterations):
        for params in tqdm(list(copy.copy(parameters))):
           
            hidden_dim, num_conv_layer, num_linear_layer, conv_type, pooling, task_type, dataset_type, batch_size = params

            experiment_result = dict()

            experiment_result['setup'] = {
                "hidden_dim": hidden_dim,
                "num_conv_layer": num_conv_layer,
                "num_linear_layers": num_linear_layer,
                "conv_type": conv_type.name.lower(),
                "pooling": pooling.name.lower(),
                "batch_size": batch_size,
                "task_type": task_type.name.lower(),
                "dataset": dataset_type.name.lower(),
            }


            logger.info(f"Training model with {experiment_result['setup']}")
            loader, dataset_characteristics = get_test_dataloader(dataset_type, task_type, batch_size)
            logger.info(f"Dataset characteristics: {dataset_characteristics}")
            

            model = GNNModel(dataset_characteristics['num_node_features'], dataset_characteristics['num_edge_features'], dataset_characteristics['num_classes'], hidden_dim, num_conv_layer, num_linear_layer, conv_type, pooling, task_type)
            
            trainer = ModelTrainer(model, dataset_characteristics, task_type)
            result = trainer.evaluate_model(loader)

            experiment_result['results'] = result
            experiment_results.append(experiment_result)
          

    # Write the dictionary to a JSON file
    with open(os.path.join(experiment_dir, 'results.json'), 'w') as file:
        json.dump(experiment_results, file, indent=4)

    flattened_data = []
    for item in experiment_results:
        flat_item = {}
        flat_item.update(item['setup'])
        flat_item.update(item['results'])
        flattened_data.append(flat_item)

    # Creating DataFrame
    df = pd.DataFrame(flattened_data)

    # Rounding float values to 2 decimal places
    df = df.round(3)

    df.to_csv(os.path.join(experiment_dir, 'results.csv'), index=True)
    
    if return_results:  
        return df


def main():
    # Define parameters
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