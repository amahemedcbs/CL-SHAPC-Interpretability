import matplotlib.pyplot as plt
import torch
import numpy as np
import scipy
import os
from jsd import convert_compare_dict_for_loading
from captum.attr import visualization as viz



algorithm = "iTAML"
dataset = "mnist"

save_path = f"analysis/{algorithm}/{dataset}/inclass_jsd.mat"
if not os.path.isfile(save_path):
    scipy.io.savemat(save_path, {})
saved_data = scipy.io.loadmat(save_path, simplify_cells=True)

# Load the saved data, if possible
keys_to_remove = ['__header__', '__version__', '__globals__']
mat_dict = {key: value for key, value in saved_data.items() if key not in keys_to_remove}



trial = 1
cls = 0
compare_dicts = [mat_dict[f"t{trial}"][f"c{cls}_maps"]["ses0"], mat_dict[f"t{trial}"][f"c{cls}_maps"]["ses4"]]
compare_dicts = convert_compare_dict_for_loading(compare_dicts)
compare_dict = compare_dicts[0]

#sample = 1
nrows, ncols = (2, 5)
fig, ax = plt.subplots(nrows, ncols, figsize=(15, 5))
for ind in range(10):
    sample = ind+10
    grads = []
    grad = compare_dict[sample]["grad"]
    original = compare_dict[sample]["original"]
    #original = (np.clip(original, 0, 1) * 255).astype(np.uint8)
    #original = np.clip(original, 0, 1)

    # Convert to H, W, C shape
    original = np.transpose(original.reshape((1,28,28)), (1, 2, 0))


    #norm_attr = viz._normalize_attr(grad, "absolute_value", reduction_axis=0)

    if isinstance(grad, np.ndarray):
        attr_combined = torch.from_numpy(grad)
        attr_combined = attr_combined.unsqueeze(0)
    #if grad.shape[0] == 3:
    #    attr_combined = torch.sum(attr_combined, dim=0)


    #plt_fig, plt_axis = plt_fig_axis

    #norm_attr = grad
    cmap = "Blues"
    vmin = 0
    vmax = 1

    #heat_map = plt.imshow(norm_attr, cmap=cmap, vmin=vmin, vmax=vmax)
    #heat_map2 = plt.imshow(attr_combined, cmap=cmap, vmin=vmin, vmax=vmax)

    grad = attr_combined.numpy()
    grad = np.transpose(grad, (1, 2, 0))
    #'''

    # Select row and column for saliency image
    if ind > 4:
        row, col = (1, ind - 5)
    else:
        row, col = (0, ind)

    '''
    #plt_fig_axis = (fig, ax[row][(2 * col)])
    plt_fig_axis = (fig, ax[row][col])
    _ = viz.visualize_image_attr(grad,
                                 original,
                                 method="original_image",
                                 sign="all",
                                 fig_size=(4, 4),
                                 plt_fig_axis=plt_fig_axis,
                                 show_colorbar=False,
                                 title="Original Image",
                                 use_pyplot=False)
    #'''

    #print("Row:",row)
    #print("Col:", col)

    #'''
    plt_fig_axis = (fig, ax[row][col])
    _ = viz.visualize_image_attr(grad,
                                 original,
                                 method="blended_heat_map",
                                 sign="absolute_value",
                                 fig_size=(4, 4),
                                 plt_fig_axis=plt_fig_axis,
                                 show_colorbar=True,
                                 title="Saliency Map",
                                 use_pyplot=False)
    #plt.show()
    #'''

    '''
    methods = ["original_image", "blended_heat_map"]
    signs = ["all", "absolute_value"]
    titles = ["Original Image", "Saliency Map"]
    colorbars = [False, True]

    # Select row and column for saliency image
    if ind > 9:
        row, col = (1, ind - 5)
    else:
        row, col = (0, ind)

    # Generate original images and saliency images
    for i in range(2):
        # print(f"Ind: {ind}\nRow: {row}\nCol: {col}\n")
        plt_fig_axis = (fig, ax[row][(2 * col) + i])
        _ = viz.visualize_image_attr(grad, original,
                                     method=methods[i],
                                     sign=signs[i],
                                     fig_size=(4, 4),
                                     plt_fig_axis=plt_fig_axis,
                                     cmap=cmap,
                                     show_colorbar=colorbars[i],
                                     title=titles[i],
                                     use_pyplot=False)

fig.tight_layout()
#plt.close()

'''
fig.tight_layout()
plt.show(block=True)
#fig.show(block=True)


