# Running SHAPC-Mean Experiments
This includes instructions for running the following experiments:
1. Loading Models??
2. Generating SHAP Values
3. Generating SHAPC Values
4. Generating SHAPC-Mean Values
5. Generating SHAPC-Mean Figures of Merit (FoMs) 

## Setup

### Clone
Clone this GitHub repository:
```
git clone https://github.com/AquaMouse247/CLSal-Analysis.git
cd CLSal-Analysis
```

### Requirements
In order to run experiments 1 - 4, a minimum of Python 3.x.x is required, along with the following packages:
```
List packages here.
```

In order to run experiments 4 and 5, a minimum of MATLAB x.x.x is required, along with the following packages:
- List packages here.

### Datasets
The code can generate results for the following datasets:
- CIFAR10
- CIFAR100
- TinyImageNet

### Algorithms
A total of nine algorithms were tested:
- `iTAML`
- `RPSNet`
- `FOSTER`
- `MEMO`
- `DER`
- `iCARL`
- `DS-AL`
- `TagFex`
- `XDER`

### Saved Models
This code relies on saved models. In order to load these models, the `load_model.py` script must be used. Inside this script, set the `filepath` parameter to the root folder for your saved models. To use a specific model, put the saved model.pth file into the `[root_model_filepath]/[algorithm]/[saved_model].pth`.

## Running Specific Experiments
## Experiment 1
Use the `blank` script.

## Experiment 2
Use the `generate_shap_values.py` script.

## Experiment 3
Use the `generate_shapc_values.py` script.

## Experiment 4
Use the `shapc_mean_values.mlx` script.

## Experiment 5
Use the `shapc_mean_values.mlx` script.
