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
- CIFAR10
- CIFAR100
- TinyImageNet

### Algorithms
A total of nine algorithms were tested:
- `iTAML`
- `RPSnet`
- `FOSTER`
- `MEMO`
- `DER`
- `iCARL`
- `DS-AL`
- `TagFex`
- `XDER`

### Saved Models
This code relies on saved models. In order to load these models, the `load_model.py` script must be used. Inside this script, set the `filepath` parameter to the root folder for your saved models. To use a specific model, put the saved model.pth file into the `[root_model_filepath]/[algorithm]/[saved_model].pth`.

### Training and Saving Models
For the nine total algorithms, four different GitHub repositories were used to train and save the models.

iTAML & RPSnet - use the https://github.com/brjathu/iTAML & https://github.com/brjathu/RPSnet repositories, respectively.

FOSTER, MEMO, DER, iCARL, DS-AL, and TagFex - use the https://github.com/LAMDA-CL/PyCIL repository.

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
For iTAML: `memory_[task].pickle`, `sample_per_task_testing_[task].pickle`, and `session_[task]_model_best.pth` for each session/task are needed.

For RPSnet: `path_[task]_0`, `fixed_path_[task]_0`, and `session_[task]_model_best.pth` for each session/task are needed.

For PyCIL and XDER: Only the `[saved_model].pth` files for each session/task are needed.

Place the needed files in the `saved_models/[alg_name]/` folder.

## Running Specific Experiments
### Adjusting Models for SHAP Value Calculation
First, some adjustments must be made to allow for SHAP calculation. These adjustments can be made after saving the models:
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

### Experiment 1
Use the `generate_shap_values.py` script to receive a 'shap_values_first_last_[samples_number].npy' file in `analysis/[algorithm]/[dataset]`.

---
### Experiment 2
**Requires Experiment 1 to be completed for the desired algorithm and dataset.**

Use the `calc_shapc_values.py` script to receive a 'shapc_values_first_last_[samples_number].mat' file in `analysis/[algorithm]/[dataset]`.

---
### Experiment 3
**Requires Experiment 2 to be completed for the desired algorithm and dataset.**

Use the `shapc_mean_values.mlx` script.

---
### Experiment 4
**Requires Experiment 3 to be completed for the desired algorithms and datasets.**

Use the `identify_scenarios.mlx` script.
