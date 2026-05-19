import matplotlib.pyplot as plt
import torch
from torch_dct import dct_2d
import numpy as np
import scipy
import os
from jsd import convert_compare_dict_for_loading
from captum.attr import visualization as viz


def zigzag_scan(matrix):
    if not isinstance(matrix, torch.Tensor):
        raise ValueError("Input must be a PyTorch tensor.")

    rows, cols = matrix.shape
    patternX = []
    patternY = []
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

    return torch.tensor(result[1:])


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
classes = [0,1]

inclass_diffs = []
ooc_diffs = []

for i in range(10):
    compare_dicts = [mat_dict[f"t{trial}"][f"c0_maps"]["ses0"], mat_dict[f"t{trial}"][f"c1_maps"]["ses0"]]
    compare_dicts = convert_compare_dict_for_loading(compare_dicts)

    #sample = 0
    sample = i

    grad1 = compare_dicts[0][sample]["grad"]
    grad2 = compare_dicts[0][sample+1]["grad"]
    grad3 = compare_dicts[1][sample]["grad"]
    grads = [grad1, grad2, grad3]
    data = np.empty(0)

    for i in range(len(grads)):
        if dataset == "mnist": grads[i] = grads[i].reshape(1, 28, 28)
        grads[i] = viz._normalize_attr(grads[i], "absolute_value", reduction_axis=0)

        if isinstance(grads[i], np.ndarray):
            grads[i] = torch.from_numpy(grads[i])

    #'''
    dcti = dct_2d(grads[0], norm='ortho')
    dctj = dct_2d(grads[1], norm='ortho')
    dctk = dct_2d(grads[2], norm='ortho')
    #'''

    '''
    dcti = dct_2d(grads[0])
    dctj = dct_2d(grads[1])
    dctk = dct_2d(grads[2])
    #'''

    zigzagi = zigzag_scan(dcti)
    zigzagj = zigzag_scan(dctj)
    zigzagk = zigzag_scan(dctk)

    inclass_diffs.append(torch.norm(dcti - dctj).item())
    ooc_diffs.append(torch.norm(dcti - dctk).item())

print("In-Class Differences")
print("----------------------")
print(inclass_diffs)
inclass_avg = np.mean(inclass_diffs).round(4)
print("Avg:", inclass_avg)

print("\nOut-of-Class Differences")
print("----------------------")
print(ooc_diffs)
ooc_avg = np.mean(ooc_diffs).round(4)
print("Avg:", ooc_avg)

if inclass_avg > ooc_avg:
    print("\n\033[1mThe \033[34min-class differences\033[0;1m are higher.\033[0m")
else:
    print("\n\033[1mThe \033[31mout-of-class differences\033[0;1m are higher.\033[0m")
