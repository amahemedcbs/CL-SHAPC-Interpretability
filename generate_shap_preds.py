### Load compare dict
import shap
import numpy as np
import scipy.io
import os
import torch

# Custom Imports
from saliency_generator import SalGenArgs, iTAMLArgs, MnistArgs
from saliency_generator import load_model, generate_predictions
import saliency_dataloader as sdl
from Saliency.imports import generate_path



algorithm = "der"
dataset = "cifar10"
num_sessions = 10 if dataset == "cifar100" or dataset == "imagenet200" else 5
device = "cuda:0" if torch.cuda.is_available() else "cpu"



if algorithm == "iTAML":
   SalGenArgs.args = iTAMLArgs
else: SalGenArgs.args = MnistArgs

if dataset == "cifar100":
    SalGenArgs.class_per_task = 10
    SalGenArgs.num_class = 100
    filepath = f"analysis/{algorithm}/{dataset}/shap_values_first_last_2000.npy"
elif dataset == "imagenet200":
    SalGenArgs.class_per_task = 20
    SalGenArgs.num_class = 200
    filepath = f"analysis/{algorithm}/{dataset}/shap_values_first_last_4000.npy"
else:
    SalGenArgs.class_per_task = 2
    SalGenArgs.num_class = 10
    filepath = f"analysis/{algorithm}/{dataset}/shap_values_first_last_1000.npy"


SalGenArgs.algorithm = algorithm
SalGenArgs.dataset = dataset
SalGenArgs.args.dataset = SalGenArgs.dataset
SalGenArgs.args.num_class = SalGenArgs.num_class
#filepath = f"analysis/noshuffle/{algorithm}/shap_values_first_last_1000.npy"


shap_values_loaded = np.load(filepath, allow_pickle=True)  # ['shap_dict']
#shap_values_loaded = np.load(f"analysis/{algorithm}/{dataset}/shap_values_first_last_1000.npy", allow_pickle=True)  # ['shap_dict']
num_imgs = len(shap_values_loaded[()].keys())
shap_dict = {}
for i in range(num_imgs):
    shap_dict[f'{i}'] = shap_values_loaded[()][f'{i}']

if dataset == "cifar100" or dataset == "imagenet200":
    samples = range(20*SalGenArgs.class_per_task*(num_sessions-1))
else:
    samples = range(100*SalGenArgs.class_per_task*(num_sessions-1))

for sample in samples:
    test_sample = shap_dict[f'{sample}']
    test_sess = list(test_sample.keys())
    test_sess.remove(test_sess[1])
    #print(test_sess)
    ses = int(test_sess[0][-1])

    # Get test dataset
    sal_dataloader = sdl.SalDataloader(SalGenArgs)
    if dataset == "cifar100":
        test_imgs, test_labels, _, STD, MEAN = sal_dataloader.load_data(range(ses * 10, (ses * 10) + 10), 20, batch_size=10000)
    else:
        test_imgs, test_labels, _, STD, MEAN = sal_dataloader.load_data([ses*2, (ses*2)+1], 100, batch_size=10000)
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
    models = [load_model(algorithm, dataset, i, args=SalGenArgs.args).to(device) for i in [int(test_sess[0][-1]),int(test_sess[-1][-1])]]

    # Generate predictions
    if algorithm == "RPSnet":
        infer_path = generate_path(ses, SalGenArgs.dataset, SalGenArgs.args)
        preds = [generate_predictions(algorithm, models[0], int(test_sess[0][-1]), test_img, infer_path=infer_path),
                 generate_predictions(algorithm, models[1], int(test_sess[-1][-1]), test_img, infer_path=infer_path)]
    else:
        preds = [generate_predictions(algorithm, models[0], int(test_sess[0][-1]), test_img),
                 generate_predictions(algorithm, models[1], int(test_sess[-1][-1]), test_img)]

    # Move preds to cpu
    preds[0] = preds[0].cpu()
    preds[1] = preds[1].cpu()

    if preds[0].item() == preds[1].item() and preds[0].item() != test_label.item():
        print("\033[1mFound one!\033[0m")
    print(f"Sample {sample}: {preds}")

    # Store predictions

    # Load the saved preds, if possible
    if os.path.isfile(f"analysis/noshuffle/{dataset}_preds.mat"):
        loaded_preds = scipy.io.loadmat(f"analysis/noshuffle/{dataset}_preds.mat", simplify_cells=True)
        keys_to_remove = ['__header__', '__version__', '__globals__']
        pred_dict = {key: value for key, value in loaded_preds.items() if key not in keys_to_remove}
    else:
        pred_dict = {}

    if f'{algorithm}' not in pred_dict: pred_dict[f'{algorithm}'] = {}
    if f'sample{sample}' not in pred_dict[f'{algorithm}']: pred_dict[f'{algorithm}'][f'sample{sample}'] = {}
    pred_dict[f'{algorithm}'][f'sample{sample}'][f'pred_{test_sess[0]}'] = preds[0].item()
    pred_dict[f'{algorithm}'][f'sample{sample}'][f'pred_{test_sess[-1]}'] = preds[1].item()

    # Save shap values to filepath
    scipy.io.savemat(f"analysis/noshuffle/{dataset}_preds.mat", pred_dict)


    '''
    import scipy.io

    # Load the saved preds, if possible
    loaded_preds = scipy.io.loadmat(f"dsal_cifar100_preds.mat", simplify_cells=True)
    keys_to_remove = ['__header__', '__version__', '__globals__']
    pred_dict = {key: value for key, value in loaded_preds.items() if key not in keys_to_remove}
    pred_dict['dsal'] = pred_dict['ds-al']
    pred_dict.pop('ds-al')
    pred_dict.keys()
    
    # Save shap values to filepath
    scipy.io.savemat(f"dsal_cifar100_preds_fixed.mat", pred_dict)
    '''