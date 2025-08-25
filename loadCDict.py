### Load compare dict
import torch
import scipy.io
import numpy as np
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


algorithm = "der"
dataset = "cifar10"
distill = True
num = 10 if dataset == "cifar100" else 5
num_imgs = 20
ses = 0
ind = 5

### Import colormap ###
colors = []
for j in np.linspace(1, 0, 100):
    colors.append((30.0 / 255, 136.0 / 255, 229.0 / 255, j))
for j in np.linspace(0, 1, 100):
    colors.append((255.0 / 255, 13.0 / 255, 87.0 / 255, j))
red_transparent_blue = LinearSegmentedColormap.from_list("red_transparent_blue", colors)
#######################
#compare_dict = torch.load(f"SaliencyMaps/{algorithm}/{dataset}/compare_dict_sess{ses}.pt")
#print(compare_dict[ind-1])

shap_values_loaded = np.load("./shap_values_first_last.npy", allow_pickle=True)#['shap_dict']
shap_dict = {}
for i in range(num_imgs):
    shap_dict[f'{i}'] = shap_values_loaded[()][f'{i}']

sample = 0
test = shap_dict[f'{sample}'][f'ses{ses}']['shap_values']
#print(shap_dict[f'{sample}'][f'ses{ses}']['shap_values'])
test_tp = np.transpose(test.squeeze(), (1, 2, 0))
im = imshow(test_tp, cmap=red_transparent_blue)
cbar = plt.colorbar(im)
plt.show()
pass