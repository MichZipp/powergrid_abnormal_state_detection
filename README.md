# Detecting Demand Not Served (DNS) and Cascading Failures (CF) in Power Grids using Graph Neural Networks

Authors: Michael Zipperle and Min Wang 

This project aims to develop an artificial intelligence-enhanced detection tool for predicting Demand Not Served (DNS) amd Cascading Failures (CF) in power grids. A deep learning model based on graph neural networks is designed for this purpose. The model predicts whether a DNS or CF will occur based on the input power grid status and topology. To generate sufficient training data for the model, we collaborate with the Reliability and Risk Engineering Group in ETH Zurich and build a physics-based simulation model for power grid analysis on the Cascades platform. While we cannot share the original dataset yet, this repository contains some testing subset which has been derived from the original dataset.

## Problem Definition

The aim of the detection tool is to distinglish between healthy and unhealthy power grid states. A healthy powergrids state is characterized by the absence of DNS and CF.

We have defined the learning task of the detection tool as a binary or multiclass classification problem.

The multiclass labels are:
- 0: DNS and CF
- 1: DNS and no CF
- 2: no DNS and CF
- 3: no DNS and no CF

The binary labels are:
- 1: DNS and CF or DNS and no CF or no DNS and CF
- 0: no DNS and no CF

## Preparation

Install all required python packages:

`pip install -r requirements.txt`

Download the testing subset and store in the `datasets` folder:

https://drive.google.com/drive/folders/1-jLgaCfTE7v3VBzfSUIfx3mK5Yli9yu1?usp=share_link

## Run experiments on the testing subset

The experiments are highly configureable:
- hidden_dims = [&lt;int&gt;, ...]
- num_conv_layers = [&lt;int&gt;, ...]
- num_linear_layers = [&lt;int&gt;, ...]
- conv_types = [&lt;ConvType&gt;,...]
- task_types = [&lt;TaskType&gt;,...]
- dataset_types = [&lt;DatasetType&gt;,...]
- batch_sizes = [&lt;int&gt;, ...]
- poolings = [&lt;PoolingType&gt;,...]

The following options are available for the costum data_types:
- ConvType: TRANSFORMER, GAT, GATv2
- TaskType: BINARY, MULTICLASS
- DatasetType: IEEE39, IEEE118
- PoolingType: MEAN, MAX, SUM, ALL

We applied grid search to find the optimal parameters and included the corresponding models in this repository
- hidden_dims = 32
- num_conv_layers = 3
- num_linear_layers = 2
- conv_types = TRANSFORMER
- task_types = N/A
- dataset_types = N/A
- batch_sizes = 128
- poolings = MEAN

The results can be replicated by running the following code either directly in python or in a notebook:

```
import test
from itertools import product
from data_structures import *

# Define parameters
number_of_iterations = 1
hidden_dims = [32]
num_conv_layers = [3]
num_linear_layers = [2]
conv_types = [ConvType.TRANSFORMER]
task_types = [TaskType.MULTICLASS, TaskType.BINARY]
dataset_types = [DatasetType.IEEE118, DatasetType.IEEE39]
batch_sizes = [128]
poolings = [PoolingType.MEAN]


# Create a grid of all parameter combinations
param_grid = product(hidden_dims, num_conv_layers, num_linear_layers, conv_types, poolings, task_types, dataset_types, batch_sizes)

test.run_experiment(param_grid, number_of_iterations, debug=False)
```

The results are automatically stored in the `experiments` folder in the root directory.

Here is an overview of the results:
| task_type  | dataset | accuracy | balanced_accuracy | sensitivity | specificity | precision | recall |  f1   |   TP   |   TN    |  FP  |  FN  |
|------------|---------|----------|-------------------|-------------|-------------|-----------|--------|-------|--------|---------|------|------|
| multiclass | ieee118 | 0.967    | 0.828             | 0.967       | 0.000       | 0.977     | 0.967  | 0.970 | NaN    | NaN     | NaN  | NaN  |
| multiclass | ieee39  | 0.968    | 0.969             | 0.968       | 0.000       | 0.981     | 0.968  | 0.972 | NaN    | NaN     | NaN  | NaN  |
| binary     | ieee118 | 0.995    | 0.994             | 0.993       | 0.995       | 0.949     | 0.993  | 0.970 | 1179.0 | 13749.0 | 64.0 | 8.0  |
| binary     | ieee39  | 0.991    | 0.995             | 1.000       | 0.990       | 0.883     | 1.000  | 0.938 | 235.0  | 3034.0  | 31.0 | 0.0  |

## Train own models and run experiments

One can create a dataset of powergrid states with binary and multiclass labels and store them in respective folders. One must give the dataset a name and extend the `DatasetType` data structure in `data_structures.py` to support it.

Similar to running experiments on the test subsets, one can train and evaluate the model by running the following code either directly in python or in a notebook:

```
import train
from itertools import product
from data_structures import *

# Define parameters
number_of_iterations = 1
hidden_dims = [32]
num_conv_layers = [3]
num_linear_layers = [2]
conv_types = [ConvType.TRANSFORMER]
task_types = [TaskType.MULTICLASS, TaskType.BINARY]
dataset_types = [DatasetType.<NEWDATASET>]
batch_sizes = [128]
poolings = [PoolingType.MEAN]


# Create a grid of all parameter combinations
param_grid = product(hidden_dims, num_conv_layers, num_linear_layers, conv_types, poolings, task_types, dataset_types, batch_sizes)

train.run_experiment(param_grid, number_of_iterations, debug=False)
```

The results are automatically stored in the `experiments` folder in the root directory.