### Load compare dict
import numpy as np
import torch
import shap
import matplotlib.pyplot as plt

# Custom Imports
from setup_args_fixed import SHAPArgs, create_shap_value_filepath, create_preds_savepath
import utils.shap_dataloader as sdl


# Select Algorithm and Dataset
algorithm = "der"
dataset = "dermamnist"
shapArgs = SHAPArgs(algorithm, dataset)


first_last_only = True
filepath = create_shap_value_filepath(shapArgs, first_last_only) + ".npy"

num_tasks = shapArgs.dataset_params.num_task
num_class = shapArgs.dataset_params.num_class
cls_per_task = shapArgs.dataset_params.class_per_task
shap_samples = shapArgs.dataset_params.shap_samples
init_cls = getattr(shapArgs.dataset_params, 'init_cls', cls_per_task)


shap_values_loaded = np.load(filepath, allow_pickle=True)
num_imgs = len(shap_values_loaded[()].keys())
shap_dict = {}
for i in range(num_imgs):
    shap_dict[f'{i}'] = shap_values_loaded[()][f'{i}']

sample = 30
test_sample = shap_dict[f'{sample}']
test_sess = list(test_sample.keys())
test_sess = [s for s in test_sess if s.startswith('ses')]
ses, last_ses = int(test_sess[0].replace('ses', '')), int(test_sess[-1].replace('ses', ''))

# Reshape/Transpose to Batch x Height x Width x Channel
if dataset == "mnist":
    test_shaps = [shap_dict[f'{sample}'][f'ses{ses}']['shap_values'].reshape(28, 28, 1),
                  shap_dict[f'{sample}'][f'ses{last_ses}']['shap_values'].reshape(28, 28, 1)]
else:
    s_first = shap_dict[f'{sample}'][f'ses{ses}']['shap_values']
    s_last = shap_dict[f'{sample}'][f'ses{last_ses}']['shap_values']
    if s_first.ndim == 5:
        s_first = s_first.squeeze(-1)
    if s_last.ndim == 5:
        s_last = s_last.squeeze(-1)
    if s_first.ndim == 4 and s_first.shape[1] in [1, 3]:
        s_first = s_first.transpose([0, 2, 3, 1])
    if s_last.ndim == 4 and s_last.shape[1] in [1, 3]:
        s_last = s_last.transpose([0, 2, 3, 1])
        
    test_shaps = [s_first, s_last]

# Get test dataset
sal_dataloader = sdl.ShapDataloader(shapArgs)

if dataset == "cifar100":
    desired_cls = list(range(ses * 10, (ses * 10) + 10))
    sal_imgs, sal_labels, _, STD, MEAN = sal_dataloader.load_data(desired_cls, 20, batch_size=10000)
elif dataset == "imagenet200":
    desired_cls = list(range(ses * 20, (ses * 20) + 20))
    sal_imgs, sal_labels, _, STD, MEAN = sal_dataloader.load_data(desired_cls, 20, batch_size=10000)
else:
    if ses == 0:
        desired_cls = list(range(init_cls))
    else:
        start_c = init_cls + (ses - 1) * cls_per_task
        end_c = init_cls + ses * cls_per_task
        desired_cls = list(range(start_c, end_c))

    sal_imgs, sal_labels, _, STD, MEAN = sal_dataloader.load_data(desired_cls, shap_samples, batch_size=10000)

print("Len of sal_imgs:", len(sal_imgs))

test_imgs, test_labels = sal_imgs, sal_labels

# Compute local index within the loaded session images
if ses == 0:
    adj_sample = sample
else:
    adj_sample = sample - (init_cls * shap_samples + (ses - 1) * cls_per_task * shap_samples)

adj_sample = min(adj_sample, len(test_imgs) - 1)

test_img = test_imgs[adj_sample]
if dataset != "mnist":
    test_img = sal_dataloader.denormalize(test_img)
test_img_np = np.transpose(test_img.numpy(), [1, 2, 0])
test_img_np = np.clip(test_img_np, 0.0, 1.0)

labels = [f'ses{ses}', f'ses{last_ses}']

plt_fig, plt_axis = plt.subplots(2, 2, figsize=(6, 6))
for ax in plt_axis[:, 0]:
    ax.imshow(test_img_np)
    ax.set_title("Original Image")
for ax in plt_axis[:, 1]:
    ax.imshow(np.mean(test_img_np, axis=2), cmap="gray")

s0 = np.sum(np.abs(test_shaps[0].squeeze()), axis=-1)
s1 = np.sum(np.abs(test_shaps[1].squeeze()), axis=-1)

vmax = max(s0.max(), s1.max())
vmin = min(s0.min(), s1.min())

hm1 = plt_axis[0, 1].imshow(s0, cmap='plasma', vmin=vmin, vmax=vmax, alpha=0.5)
plt_axis[0, 1].set_title(labels[0])

hm2 = plt_axis[1, 1].imshow(s1, cmap='plasma', vmin=vmin, vmax=vmax, alpha=0.5)
plt_axis[1, 1].set_title(labels[1])

for ax in plt_axis.flat:
    ax.set_xticks([])
    ax.set_yticks([])

cbar = plt_fig.colorbar(hm2, ax=plt_axis[:, 1], orientation='vertical', pad=0.04)
cbar.ax.set_title("Most Important", pad=8)
cbar.ax.set_xlabel("Least Important", labelpad=8)

plt.subplots_adjust(right=0.82)
import os

# Define your Google Drive save directory
save_dir = f"/content/drive/MyDrive/CL-SHAPC-Interpretability/shap_visuals/{algorithm}/{dataset}"
os.makedirs(save_dir, exist_ok=True)

save_path = os.path.join(save_dir, f"shap_sample_{sample}.png")

# Save high-resolution figure before showing
plt.savefig(save_path, bbox_inches='tight', dpi=300)
print(f"Saved plot to: {save_path}")

plt.show()
