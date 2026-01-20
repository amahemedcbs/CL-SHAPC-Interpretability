# Improved Efficiency and Class Dependency Examination with SHAPC
## Introduction
This repository contains code for an efficient SHAPC calculation as well as examiniation into the nature of class dependence in the SHAPC calculation.

## Results
## How To Use
### Clone
Clone this GitHub repository:
```
git clone https://github.com/AquaMouse247/CLSal-Analysis.git
cd CLSal-Analysis
```
<!--### Dependencies
The following Python libraries are required:
run `pip install -r requirements.txt` to install all dependencies.
-->

The shapTest.py and calcSHAPC.py files are used to calculate the SHAP values and the SHAPC values for the images, respectively.

The shapcMeanVar.mlx and shapcCompare.mlx scripts in the analysis folder are then used to calculate and compare the SHAPC-Mean of each algorithm.


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