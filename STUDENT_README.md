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
In order to run experiments 1 - 4, a minimum of Python 3.10.0 is required, along with the following packages:
```
kornia==0.8.2
numpy==2.4.4
onedrivedownloader==1.1.3
Pillow==12.2.0
quadprog==0.1.13
scikit_learn==1.8.0
scipy==1.17.1
shap==0.48.0
torch==2.7.0
torchvision==0.22.0
tqdm==4.66.5
wandb==0.26.1
```

In order to run experiments 4 and 5, a minimum of MATLAB R2024b is required, along with the following packages:

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
Use the `calc_shapc_values.py` script.

## Experiment 4
Use the `shapc_mean_values.mlx` script.

## Experiment 5
Use the `identify_scenarios.mlx` script.
