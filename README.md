# Improved Efficiency and Class Dependency Examination with SHAPC
## Introduction
This repository contains code for three optimized approaches to computing SHAPC-Mean that reduce samples, task-pair iterations, or a combination of both. This decreases overall computation time while maintaining metric performance. Additionally, it contains code for calculating a classwise SHAPC-Mean metric to assess potential class-dependent biases.

## Results
## How To Use
### Clone
Clone this GitHub repository:
```
git clone https://github.com/AquaMouse247/CL-SHAPC-Interpretability.git
cd CL-SHAPC-Interpretability
```
<!--### Dependencies
The following Python libraries are required:
run `pip install -r requirements.txt` to install all dependencies.
-->

The 'generate_shap_values.py' and 'generate_shapc_values.py' files are used to calculate the SHAP values and the SHAPC values for the images, respectively.

The 'shapcMeanVar.mlx' and 'shapcFinalCompare.mlx' scripts in the analysis folder are then used to calculate and compare the SHAPC-Mean of each algorithm.


### Datasets
The code can generate results for the following datasets:
- CIFAR10
- CIFAR100

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
<!-- Additional algorithms can be added by...-->
## Acknowledgments
## Citation
