### Load compare dict
import numpy as np
import scipy.io
import os
import torch

# Custom Imports
from utils.setup_args import SHAPArgs, create_shap_value_filepath
from utils.load_models import load_model, load_meta_models, generate_predictions
from utils.model_parameters import pycil_algs

import utils.shap_dataloader as sdl
from models.RPSnet.rps_net import generate_path



algorithm = "der"
dataset = "cifar10"
shapArgs = SHAPArgs(algorithm, dataset)
device = "cuda:0" if torch.cuda.is_available() else "cpu"



first_last_only = True
filepath = create_shap_value_filepath(shapArgs, first_last_only)

num_tasks = shapArgs.dataset_params.num_task
cls_per_task = shapArgs.dataset_params.class_per_task


shap_values_loaded = np.load(filepath, allow_pickle=True)  # ['shap_dict']
num_imgs = len(shap_values_loaded[()].keys())
shap_dict = {}
for i in range(num_imgs):
    shap_dict[f'{i}'] = shap_values_loaded[()][f'{i}']

samples = range(shapArgs.dataset_params.shap_samples*cls_per_task*(num_tasks-1))

for sample in samples:
    test_sample = shap_dict[f'{sample}']
    test_sess = list(test_sample.keys())
    test_sess.remove(test_sess[1])
    #print(test_sess)
    ses = int(test_sess[0][-1])

    # Get test dataset
    sal_dataloader = sdl.ShapDataloader(shapArgs)
    if dataset == "cifar100":
        test_imgs, test_labels, _, STD, MEAN = sal_dataloader.load_data(range(ses * 10, (ses * 10) + 10), 20, batch_size=10000)
    elif dataset == "imagenet200":
        test_imgs, test_labels, _, STD, MEAN = sal_dataloader.load_data(range(ses * 20, (ses * 20) + 20), 20, batch_size=10000)
    else:
        test_imgs, test_labels, _, STD, MEAN = sal_dataloader.load_data([ses * 2, (ses * 2) + 1], 100, batch_size=10000)
    #print("Len of sal_imgs:", len(test_imgs))
    test_imgs, test_labels = test_imgs.to(device), test_labels.to(device)

    if dataset == "cifar100":
        sample_multiplier = 10 * 20
    elif dataset == "imagenet200":
        sample_multiplier = 20 * 20
    else:
        sample_multiplier = 2 * 100

    # Get test image
    test_img = test_imgs[sample-(ses*sample_multiplier)].unsqueeze(0)
    test_label = test_labels[sample-(ses*sample_multiplier)]
    models = [load_model(algorithm, dataset, i, shapArgs=shapArgs).to(device) for i in [int(test_sess[0][-1]),int(test_sess[-1][-1])]]

    # Generate predictions
    if algorithm == "RPSnet":
        infer_path = generate_path(ses, dataset, shapArgs.algorithm_args)
        preds = [generate_predictions(algorithm, models[0], int(test_sess[0][-1]), test_img, infer_path=infer_path, cls_per_task=cls_per_task),
                 generate_predictions(algorithm, models[1], int(test_sess[-1][-1]), test_img, infer_path=infer_path, cls_per_task=cls_per_task)]
    else:
        preds = [generate_predictions(algorithm, models[0], int(test_sess[0][-1]), test_img, cls_per_task=cls_per_task),
                 generate_predictions(algorithm, models[1], int(test_sess[-1][-1]), test_img, cls_per_task=cls_per_task)]

    # Move preds to cpu
    preds[0] = preds[0].cpu()
    preds[1] = preds[1].cpu()

    if preds[0].item() == preds[1].item() and preds[0].item() != test_label.item():
        print("\033[1mFound one!\033[0m")
    print(f"Sample {sample}: {preds}")

    # Store predictions

    # Load the saved preds, if possible
    if os.path.isfile(f"analysis/preds/{algorithm}_{dataset}_preds.mat"):
        loaded_preds = scipy.io.loadmat(f"analysis/preds/{algorithm}_{dataset}_preds.mat", simplify_cells=True)
        keys_to_remove = ['__header__', '__version__', '__globals__']
        pred_dict = {key: value for key, value in loaded_preds.items() if key not in keys_to_remove}
    else:
        pred_dict = {}

    if f'{algorithm}' not in pred_dict: pred_dict[f'{algorithm}'] = {}
    if f'sample{sample}' not in pred_dict[f'{algorithm}']: pred_dict[f'{algorithm}'][f'sample{sample}'] = {}
    pred_dict[f'{algorithm}'][f'sample{sample}'][f'pred_{test_sess[0]}'] = preds[0].item()
    pred_dict[f'{algorithm}'][f'sample{sample}'][f'pred_{test_sess[-1]}'] = preds[1].item()

    # Save shap values to filepath
    scipy.io.savemat(f"analysis/preds/{algorithm}_{dataset}_preds.mat", pred_dict)
