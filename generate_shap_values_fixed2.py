import os
import sys
import numpy as np
import shap
import torch
from torch.utils.data import Subset
from tqdm import tqdm
import torch.nn as nn

# Custom Imports
from setup_args_fixed import SHAPArgs, create_shap_value_filepath
from utils.load_models_copy import load_model, load_meta_models
from utils.model_parameters import pycil_algs

import utils.shap_dataloader as sdl
from models.RPSnet.rps_net import generate_path

algorithm = "tagfex"
dataset = "dermamnist"
shapArgs = SHAPArgs(algorithm, dataset)

'''
# Used when running with a shell script.
algorithm = sys.argv[1]
dataset = sys.argv[2]
#'''

first_last_only = True

# --- Set and create exact Google Drive directory path ---
filepath = create_shap_value_filepath(shapArgs, first_last_only)
drive_save_dir = os.path.dirname(filepath)
os.makedirs(drive_save_dir, exist_ok=True)
print(f"Saving output to: {filepath}.npy")

print(f"Alg: {algorithm}\nDataset: {dataset}\nFirst/Last: {first_last_only}")
num_tasks = getattr(shapArgs.dataset_params, 'num_task', getattr(shapArgs.dataset_params, 'num_tasks', 3))
cls_per_task = getattr(shapArgs.dataset_params, 'class_per_task', getattr(shapArgs.dataset_params, 'cls_per_task', 2))
init_cls = getattr(shapArgs.dataset_params, 'init_cls', cls_per_task)

# Configures models depending on the algorithm chosen
if algorithm == "iTAML":
    models = []
    for i in range(num_tasks):
        models.extend(load_meta_models(dataset, i))
    for model in models: model.set_saliency(True)

else:
    models = [load_model(algorithm, dataset, i, shapArgs=shapArgs) for i in range(num_tasks)]
    if algorithm == "RPSnet":
        infer_paths = []
        lasts = [-1] * len(models)
        for i in range(len(models)):
            infer_paths.append(generate_path(i, dataset, shapArgs.algorithm_args))
            models[i].shap_path = infer_paths[i]
            models[i].set_shap(True)
    elif algorithm in pycil_algs:
        for model in models: model.set_shap(True)
    elif algorithm == "xder":
        pass

sal_dataloader = sdl.ShapDataloader(shapArgs)

# Get train dataset
train_set = sal_dataloader.get_shap_train_set(dataset)

# Get test dataset
for i in range(num_tasks):
    if dataset == "cifar100":
        sal_imgs, sal_labels, _, STD, MEAN = sal_dataloader.load_data(range(i * 10, (i * 10) + 10), 20, batch_size=10000)
    elif dataset == "imagenet200":
        sal_imgs, sal_labels, _, STD, MEAN = sal_dataloader.load_data(range(i * 20, (i * 20) + 20), 20, batch_size=10000)
    else:
        if i == 0:
            cls_range = range(0, init_cls)
        else:
            cls_range = range(init_cls + (i - 1) * cls_per_task, init_cls + i * cls_per_task)
            
        sal_imgs, sal_labels, _, STD, MEAN = sal_dataloader.load_data(cls_range,
                                                                      shapArgs.dataset_params.shap_samples, 
                                                                      batch_size=10000)
    print("Len of sal_imgs:", len(sal_imgs))
    if i == 0:
        test_imgs, test_labels = sal_imgs, sal_labels
    else:
        test_imgs = torch.cat((test_imgs, sal_imgs), 0)
        test_labels = torch.cat((test_labels, sal_labels), 0)

# Reshape MNIST test images for RPSnet
if algorithm == "RPSnet" and dataset == "mnist":
    test_imgs = test_imgs.detach().numpy().reshape(-1, 784)
    test_imgs = torch.from_numpy(test_imgs)

# 1. Generate 100 random indices
indices = np.random.choice(len(train_set), 100, replace=False)

# 2. Create a subset of the dataset
small_train_set = Subset(train_set, indices)

# Get background data for shap explainer
if algorithm == "RPSnet" and dataset == "mnist":
    background_data = torch.cat([x[0].unsqueeze(0).reshape(-1, 784) for x in small_train_set])
else:
    background_data = torch.cat([img.unsqueeze(0) for (*_, img, lbl) in small_train_set])
background_data = shap.sample(background_data, 100)


# ==============================================================================
# Enable Full Autograd Graph for DS-AL & Analytic Models
# ==============================================================================
for model in models:
    model.eval()  # Keep eval mode for batchnorm/dropout
    model.set_shap(True)

    # 1. Enable gradients across all standard parameters
    for param in model.parameters():
        param.requires_grad_(True)

    # 2. Specifically unfreeze backbone convnet
    if hasattr(model, 'convnet'):
        for p in model.convnet.parameters():
            p.requires_grad_(True)

    # 3. Enable gradients for RandomBuffer projection matrix
    if hasattr(model, 'buffer'):
        for attr in ['W', 'weight']:
            if hasattr(model.buffer, attr):
                val = getattr(model.buffer, attr)
                if isinstance(val, torch.Tensor):
                    val.requires_grad_(True)

    # 4. Enable gradients for RecursiveLinear analytic weights
    for fc_attr in ['fc', 'fc_comp']:
        if hasattr(model, fc_attr):
            fc_obj = getattr(model, fc_attr)
            if hasattr(fc_obj, 'weight') and isinstance(fc_obj.weight, torch.Tensor):
                fc_obj.weight.requires_grad_(True)

def prepare_model_for_shap(model):
    model.eval()
    if hasattr(model, "set_shap"):
        model.set_shap(True)
    
    # 1. Unfreeze all standard parameters & convert to float32
    for p in model.parameters():
        p.requires_grad_(True)
        
    # 2. Specifically convert RandomBuffer projection matrix to active nn.Parameter
    if hasattr(model, "buffer"):
        for attr in ["W", "weight"]:
            if hasattr(model.buffer, attr):
                val = getattr(model.buffer, attr)
                if isinstance(val, torch.Tensor):
                    # Convert to float32 parameter with grad enabled
                    param = nn.Parameter(val.detach().to(torch.float32), requires_grad=True)
                    setattr(model.buffer, attr, param)
                    
    # 3. Specifically convert RecursiveLinear weights (main & comp) to active nn.Parameter
    for fc_attr in ["fc", "fc_comp"]:
        if hasattr(model, fc_attr):
            fc_obj = getattr(model, fc_attr)
            if hasattr(fc_obj, "weight") and isinstance(fc_obj.weight, torch.Tensor):
                param = nn.Parameter(fc_obj.weight.detach().to(torch.float32), requires_grad=True)
                fc_obj.weight = param

    # 4. Set entire model to float32
    model.to(torch.float32)
    return model

# Prepare all loaded session models
models = [prepare_model_for_shap(m) for m in models]


# Now instantiate explainers
explainers = [shap.GradientExplainer(models[i], background_data) for i in range(len(models))]
                    

# Create the explainer for each model
explainers = [shap.GradientExplainer(models[i], background_data) for i in range(len(models))]

shap_dict = {}

def get_sample_task(lbl, init_c, inc_c):
    if lbl < init_c:
        return 0
    return 1 + (lbl - init_c) // inc_c

for sample in tqdm(range(len(test_imgs)), desc="Progress"):
    shap_dict[f'{sample}'] = {}
    lbl = test_labels[sample].item() if isinstance(test_labels[sample], torch.Tensor) else int(test_labels[sample])
    sample_task = get_sample_task(lbl, init_cls, cls_per_task)

    for e in range(len(explainers)):
        if first_last_only:
            if algorithm == "iTAML":
                boolean_statement = (e == sample_task or e == num_tasks - 1 + sample_task)
            else:
                boolean_statement = (e == sample_task or e == num_tasks - 1)
        else:
            boolean_statement = sample_task <= e

        if boolean_statement:
            shap_values, indexes = explainers[e].shap_values(test_imgs[sample].unsqueeze(0), ranked_outputs=1)
            shap_dict[f'{sample}'][f'ses{e}'] = {'shap_values': shap_values, 'idxs': indexes}
            shap_dict[f'{sample}']['true_label'] = lbl
        else:
            continue

    # Intermittent saving to Google Drive every 50 samples or in case of interruption
    if sample % 50 == 0:
        np.save(filepath, shap_dict)

# Final save directly to Google Drive
np.save(filepath, shap_dict)
print(f"[SUCCESS] Finished! SHAP values saved to: {filepath}.npy")
