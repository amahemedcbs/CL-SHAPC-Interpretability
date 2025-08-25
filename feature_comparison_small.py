import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np

# Saliency Imports
from captum.attr import Saliency
from captum.attr import visualization as viz
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch_dct as dct

patternX = []
patternY = []


def zigzag_scan(matrix):
    if not isinstance(matrix, torch.Tensor):
        raise ValueError("Input must be a PyTorch tensor.")

    rows, cols = matrix.shape
    result = []

    for d in range(rows + cols - 1):
        # Determine the starting row and column for this diagonal
        if d % 2 == 0:  # Even diagonals (going upwards)
            row = min(d, rows - 1)
            col = d - row
            while row >= 0 and col < cols:
                patternX.append(col)
                patternY.append(row)
                result.append(matrix[row, col].item())
                row -= 1
                col += 1
        else:  # Odd diagonals (going downwards)
            col = min(d, cols - 1)
            row = d - col
            while col >= 0 and row < rows:
                patternX.append(col)
                patternY.append(row)
                result.append(matrix[row, col].item())
                row += 1
                col -= 1

    return torch.tensor(result)

def get_indices(dataset, class_name):
    indices = []
    for i in range(len(dataset.targets)):
        for j in class_name:
            if dataset.targets[i] == j:
                indices.append(i)
    return indices

def create_blended_img(image, grads, truth=True):

    original_image = np.transpose(image, (1, 2, 0))

    #print(grads.shape)
    if grads.shape[1] == 3:
        grads = np.transpose(grads.squeeze().cpu().detach().numpy(), (1, 2, 0))
    else:
        grads = np.transpose(grads.unsqueeze(0).numpy(), (1, 2, 0))

    plt.figure()
    fig, _ = viz.visualize_image_attr(grads, original_image,
                                 method="blended_heat_map",
                                 sign="absolute_value",
                                 fig_size=(0.33, 0.33),
                                 cmap="Blues" if truth else "Reds",
                                 use_pyplot=False)

    fig.savefig("TempFig.png")
    plt.close()
    blended_arr = plt.imread("TempFig.png")
    os.remove("TempFig.png")

    return blended_arr

def create_difference_data(compare_dict):
    grads = [compare_dict[i]["grad"].squeeze() for i in range(4,8)]
    if grads[0].shape[0] == 3:
        for i in range(len(grads)): grads[i] = torch.sum(grads[i], dim=0)
    size = len(grads)
    data = np.zeros((size, size), dtype=np.float64)
    #print("Before:\n",data)

    for i in range(size):
        for j in range(size):
            dcti = dct.dct_2d(grads[i])
            dctj = dct.dct_2d(grads[j])
            zigzagi = zigzag_scan(dcti)
            zigzagj = zigzag_scan(dctj)
            diff = torch.norm(dcti - dctj)
            data[i][j] = diff.item()
    #print("After:\n", data)
    return data


def create_colormap(colormap, data, ses):
    # Get the 'coolwarm' colormap
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    cmap = cm.get_cmap(colormap)

    cmapData = data[data != 0.0]
    if ses == 0: vmin, vmax = cmapData.min(), cmapData.max()

    # Define the boundaries and assign pink color to values below 100
    num_bins = 256
    boundaries = np.concatenate([[0], np.linspace(vmin, vmax, num_bins)])

    # Create a new colormap that is similar to 'coolwarm' but with the color at 0 as transparent
    colors = cmap(np.linspace(0, 1, cmap.N))  # Get the existing colors
    colors[0] = (1, 0.75, 0.8, 1)  # Set the color at 0 to be pink (RGB + alpha)

    # Create the new colormap
    new_colormap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(boundaries, len(colors))
    return new_colormap, norm


def create_diff_heatmap(algorithm, dataset, ses, compare_dict, vmin = 0, vmax = 350):
    data = create_difference_data(compare_dict)
    size = len(compare_dict)
    figsize = 7.0 + ((num_imgs+1)*10) / 100.0
    plt.figure(figsize=(figsize, figsize))
    #cmap, norm = create_colormap("coolwarm", data, ses)
    #plt.imshow(data, cmap=cmap, norm=norm, extent=[0, size, 0, size])
    plt.imshow(data, cmap="coolwarm", vmin=vmin, vmax=vmax, extent=[0, size, 0, size])

    # Add numbers to plot
    for i in range(size):
        for j in range(size):
            plt.annotate('{:.1f}'.format(data[size-1-i][j]), xy=(j+0.5, i+0.5), ha='center', va='center', color='black')

    # Change ticks to images
    # set ticks where your images will be
    ax = plt.gca()
    ticks = [0.5, 1.5, 2.5, 3.5]
    imgTicks = [create_blended_img(compare_dict[i]["original"], compare_dict[i]["grad"], compare_dict[i]["pred"]) for i in range(4,8)]
    ax.get_xaxis().set_ticks(ticks)
    ax.get_yaxis().set_ticks(ticks)
    # remove tick labels
    ax.get_xaxis().set_ticklabels([])
    ax.get_yaxis().set_ticklabels([])

    # Create x-ticks
    labels = ['Frog 1', 'Frog 2', 'Car 1', 'Car 2']
    for i in range(len(ticks)):
        #img = np.transpose(imgTicks[i].detach().numpy(), (1, 2, 0))
        img = imgTicks[i]
        imagebox = OffsetImage(img, zoom=2.5)
        imagebox.image.axes = ax

        ab = AnnotationBbox(imagebox, (ticks[i], 0),
                            xybox=(0, -7),
                            xycoords=("data", "axes fraction"),
                            boxcoords="offset points",
                            box_alignment=(.5, 1),
                            bboxprops={"edgecolor": "none"},
                            clip_on=False)

        ax.add_artist(ab)
        # Create the text label below the image
        text = labels[i]  # Replace with your desired label
        ax.text(ticks[i], -1.1, text, ha='center', va='center', fontsize=10, color='black')

    # Create y-ticks
    for i in range(len(ticks)):
        img = imgTicks[len(ticks)-1-i]
        imagebox = OffsetImage(img, zoom=2.5)
        imagebox.image.axes = ax

        ab = AnnotationBbox(imagebox, (0, ticks[i]),
                            xybox=(-7, 0),
                            xycoords=("axes fraction", "data"),
                            boxcoords="offset points",
                            box_alignment=(1, .5),
                            bboxprops={"edgecolor": "none"},
                            clip_on=False)

        ax.add_artist(ab)
        # Create the text label below the image
        text = labels[-i-1]  # Replace with your desired label
        ax.text(-1.1, ticks[i], text, ha='center', va='center', fontsize=10, color='black')


    stdDev = np.std(data)
    print("Std. Dev:", stdDev)

    if ses == 0: vmin, vmax = data.min(), data.max()

    plt.clim(vmin=vmin, vmax=vmax)
    #plt.clim(vmin=0, vmax=11000)
    #plt.clim(vmin=data.min(), vmax=data.max())
    cbar = plt.colorbar(fraction=0.046, pad=0.04)


    plt.title(f'Attention Comparison Test', horizontalalignment='center', fontsize=16)
    plt.figtext(0.55, 0.035, f'Standard Deviation: {stdDev:.2f}', ha='center', fontsize=12)
    plt.tight_layout()
    plt.savefig("attnTest.png")
    #plt.show()

    return vmin, vmax





algorithm = "iTAML"
dataset = "mnist"
distill = True
num = 10 if dataset == "cifar100" else 5
num_imgs = 10

compare_dict = torch.load(f"compare_dict_sess.pt", map_location=torch.device('cpu'))
create_diff_heatmap(algorithm, dataset, 0, compare_dict)
