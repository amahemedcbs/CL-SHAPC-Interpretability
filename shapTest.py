import torchvision.datasets
import numpy as np
import json
import shap
import sys
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tqdm import tqdm

# Custom Imports
from saliency_generator import load_model
from load_meta_model import load_meta_models

from saliency_generator import SalGenArgs, iTAMLArgs, MnistArgs
import saliency_dataloader as sdl
from rps_net import generate_path


def setup_args(algorithm, dataset) -> None:
    if algorithm == "iTAML":
       SalGenArgs.args = iTAMLArgs
    else: SalGenArgs.args = MnistArgs

    if dataset == "cifar100":
        SalGenArgs.class_per_task = 10
        SalGenArgs.num_class = 100
    elif dataset == "imagenet200":
        SalGenArgs.class_per_task = 20
        SalGenArgs.num_class = 200
    else:
        SalGenArgs.class_per_task = 2
        SalGenArgs.num_class = 10

    SalGenArgs.algorithm = algorithm
    SalGenArgs.dataset = dataset
    SalGenArgs.args.dataset = SalGenArgs.dataset
    SalGenArgs.args.num_class = SalGenArgs.num_class


algorithm = "der"
dataset = "imagenet200"

'''
algorithm = sys.argv[1]
dataset = sys.argv[2]
#'''

first_last_only = True
if first_last_only:
    filepath = f"analysis/{algorithm}/{dataset}/shap_values_first_last_2000"
else:
    filepath = f"analysis/{algorithm}/{dataset}/shap_values_full_1000"

if not os.path.isdir(f"analysis/{algorithm}/{dataset}"):
    os.makedirs(f"analysis/{algorithm}/{dataset}")


setup_args(algorithm,dataset)
print(f"Alg: {algorithm}\nDataset: {dataset}\nFirst/Last: {first_last_only}")
num_tasks = int(SalGenArgs.num_class / SalGenArgs.class_per_task)
cls_per_task = SalGenArgs.class_per_task


pycil_algs = ["der", "foster", "memo", "icarl", "simplecil", "ds-al", "tagfex"]

# Configures models depending on the algorithm chosen
if algorithm == "iTAML":
    models = []
    for i in range(num_tasks):
        models.extend(load_meta_models(dataset, i))
    for model in models: model.set_saliency(True)

else:
    models = [load_model(algorithm, dataset, i, args=SalGenArgs.args) for i in range(num_tasks)]
    if algorithm == "RPSnet":
        infer_paths = []
        lasts = [-1] * len(models)
        for i in range(len(models)):
            infer_paths.append(generate_path(i, SalGenArgs.dataset, SalGenArgs.args))
            models[i].shap_path = infer_paths[i]
            models[i].set_shap(True)
    elif algorithm in pycil_algs:
        for model in models: model.set_shap(True)
    elif algorithm == "xder":
        #for model in models: model.set_shap(True)
        pass


sal_dataloader = sdl.SalDataloader(SalGenArgs)

# Get train dataset
train_set = sal_dataloader.get_shap_train_set(dataset)

# Get test dataset
for i in range(num_tasks):
    if dataset == "cifar100":
        sal_imgs, sal_labels, _, STD, MEAN = sal_dataloader.load_data(range(i * 10, (i * 10) + 10), 20, batch_size=10000)
    elif dataset == "imagenet200":
        sal_imgs, sal_labels, _, STD, MEAN = sal_dataloader.load_data(range(i * 20, (i * 20) + 20), 20,
                                                                      batch_size=10000)
    else:
        sal_imgs, sal_labels, _, STD, MEAN = sal_dataloader.load_data([i*2, (i*2)+1], 100, batch_size=10000)
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
    background_data = torch.cat([x[0].unsqueeze(0).reshape(-1, 784) for x in small_train_set])  # Getting a sample
else:
    background_data = torch.cat([img.unsqueeze(0) for (*_, img, lbl) in small_train_set])  # Getting a sample
background_data = shap.sample(background_data, 100) # Taking 100 random samples

#Create the explainer for each model
explainers = [shap.GradientExplainer(models[i], background_data) for i in range(len(models))]

shap_dict = {}
all_shap_values = []
all_indexes = []
sample = 0

for sample in tqdm(range(len(test_imgs)), desc="Progress"):
    shap_dict[f'{sample}'] = {}
    sample_task = test_labels[sample] // cls_per_task
    for e in range(len(explainers)):
        if first_last_only:
            if algorithm == "iTAML": boolean_statement = (e == sample_task or e == num_tasks-1+sample_task)
            else: boolean_statement = (e == sample_task or e == num_tasks-1)
        else:
            boolean_statement = test_labels[sample] // cls_per_task <= e

        if boolean_statement:
            #print("Shape:", test_imgs.shape)
            #print("Sample Shape:", test_imgs[sample].unsqueeze(0).shape)
            shap_values, indexes = explainers[e].shap_values(test_imgs[sample].unsqueeze(0), ranked_outputs=1)
            shap_dict[f'{sample}'][f'ses{e}'] = {'shap_values': shap_values, 'idxs': indexes}
            shap_dict[f'{sample}']['true_label'] = test_labels[sample].item()
        else:
            continue

# Save shap values to filepath
np.save(filepath, shap_dict)
