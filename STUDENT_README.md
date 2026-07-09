# Running SHAPC-Mean Experiments
This includes instructions for running the following experiments:
1. Generating SHAP Values
2. Generating SHAPC Values
3. Generating SHAPC-Mean Values
4. Generating SHAPC-Mean Figures of Merit (FoMs) 

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

In order to run experiments 4 and 5, a minimum of MATLAB R2024b is required.

### Datasets
The code can generate results for the following datasets:
- CIFAR10 (referred to as 'cifar10' in setup files)
- CIFAR100 (referred to as 'cifar100' in setup files)
- TinyImageNet (referred to as 'imagenet200' in setup files)

### Algorithms
A total of nine algorithms were tested:
- `iTAML` (referred to as 'iTAML' in setup files)
- `RPSnet` (referred to as 'RPSnet' in setup files)
- `FOSTER` (referred to as 'foster' in setup files)
- `MEMO` (referred to as 'memo' in setup files)
- `DER` (referred to as 'der' in setup files)
- `iCARL` (referred to as 'icarl' in setup files)
- `DS-AL` (referred to as 'dsal' in setup files)
- `TagFex` (referred to as 'tagfex' in setup files)
- `XDER` (referred to as 'xder' in setup files)

### Saved Models
This code relies on saved models. In order to load these models, the `load_model.py` script must be used. Inside this script, set the `filepath` parameter to the root folder for your saved models. To use a specific model, put the saved model.pth file into the `[root_model_filepath]/[algorithm]/[saved_model].pth`.

### Training and Saving Models
For the nine total algorithms, four different GitHub repositories were used to train and save the models.

iTAML & RPSnet - use the https://github.com/brjathu/iTAML & https://github.com/brjathu/RPSnet repositories, respectively.

FOSTER, MEMO, DER, iCARL, DS-AL, and TagFex - use the https://github.com/LAMDA-CL/PyCIL repository.

Be sure to set `shuffle=False` in the experiment file for the desired algorithm.

**Note: Add the code indicated below before training in order to save the models.**
```
def _train(args):

    init_cls = 0

    # _train code continues 
    for task in range(data_manager.nb_tasks):
        logging.info("All params: {}".format(count_parameters(model._network)))
        logging.info(
            "Trainable params: {}".format(count_parameters(model._network, True))
        )
        model.incremental_train(data_manager)
        cnn_accy, nme_accy = model.eval_task()
        
        # ADD THIS: Save model after session for shap
        savemodelname = "savedmodels/{}/{}/{}_ses_{}.pth".format(
            args["model_name"],
            args["dataset"],
            args["model_name"],
            task
        )
        torch.save(model._network.state_dict(), savemodelname)

        # _train code continues

        # end of for loop

    # ADD THIS right outside of for loop
    torch.save(model._network.state_dict(), "savedmodels/{}/{}/{}_ses_{}.pth".format(args["model_name"],
                                                                                     args["dataset"],
                                                                                     args["model_name"],
                                                                                     task))
```

XDER - uses the https://github.com/aimagelab/mammoth repository.

The code must be run with the `--savecheck task` flag in order for the models to be saved.

### Adding Saved Models to Folder
For **iTAML**, the following files are needed from the :
- `memory_[task].pickle`
- `sample_per_task_testing_[task].pickle`
- `session_[task]_model_best.pth`

**Note: One of each file is required for each session/task.**
***

For **RPSnet**, the following files are needed from the:
- `path_[task]_0`
- `fixed_path_[task]_0`
- `session_[task]_model_best.pth`

**Note: One of each file is required for each session/task.**
***

For **PyCIL** and **XDER**, the following files are needed from the :
- Only the `[saved_model].pth` files for each session/task are needed. These files should be located in the `savedmodels/[alg_name]/[dataset]` of your cloned PyCIL repository.

**Note: One of each file is required for each session/task.**

Place the needed files in the `saved_models/[alg_name]/[dataset]` folder of this repository.
***

## Running Specific Experiments
<!--
### Adjusting Models for SHAP Value Calculation
Some adjustments must be made to allow for SHAP calculation. These adjustments can be made after saving the models:
For iTAML & RPSnet: Make the following adjustments in the `basic_net.py` script.
```
class BasicNet1(nn.Module):

    def __init__(
        self, args, use_bias=False, init="kaiming", use_multi_fc=False, device=None
    ):
        # Init code
        # After last 'self.var = value' in init
        self.shap = False

    def forward(self, x):
        x1, x2 = self.convnet(x)
        # Make this change
        if self.shap: return x1
        else: return x1, x2

    # Add this function
    def set_shap(self, mode):
        self.shap = mode

    @property
    def features_dim(self):
        return self.convnet.out_dim
    # Code continues
```

For PyCIL Algs: Add the following function in `inc_net.py` for each model class used (total of 6).
```
    def __init__(self, args, pretrained):
        # Init code
        # After last 'self.var = value' in init
        self.shap = False

    # Other function code continues

    # Add set_shap function above the forward function
    def set_shap(self, mode):
        self.shap = mode

    def forward(self, x):
      # Code continues
```

For XDER: No changes are needed.

The load_models.py script should now work as expected.
---
-->

### Experiment 1
Use the `generate_shap_values.py` script.

You should receive a `shap_values_first_last_[samples_number].npy` file in `analysis/[algorithm]/[dataset]`.
***

### Experiment 2
**Requires Experiment 1 to be completed for the desired algorithm and dataset.**

Use the `generate_shapc_values.py` script.

You should receive a `shapc_values_first_last_[samples_number].mat` file in `analysis/[algorithm]/[dataset]`.
***

### Experiment 3
**Requires Experiment 2 to be completed for the desired algorithm and dataset.**

Use the `shapc_mean_values.mlx` script.

You should see the SHAPC-Mean value output in the console.
***

### Experiment 4
**Requires Experiment 2 to be completed for the desired algorithms and datasets.**

1. Run the `generate_shap_preds.py` script to get the predictions for the associated SHAP images for each algorithm and dataset to receive a `[algorithm]_[dataset]_preds.mat` file in `analysis/preds/[algorithm]/[dataset]`.
2. Run the `merge_preds.mlx` script to generate/update the `cifar10_preds.mat` file for each desired algorithm and dataset.
3. Use the `identify_scenarios.mlx` script to receive the FoM tables.
