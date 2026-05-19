import torch
import numpy as np
from torch_dct import dct_2d
from feature_comparison import zigzag_scan
from scipy.io import savemat
from captum.attr import Saliency
from captum.attr import visualization as viz
import os

from saliency_generator import load_model, generate_predictions
from saliency_generator import SalGenArgs, iTAMLArgs, MnistArgs
import saliency_dataloader as sdl
from rps_net import generate_path


def generate_XA(compare_dict):
    grads = [compare_dict[i]["grad"] for i in range(len(compare_dict))]
    #if grads[0].shape[0] == 3:
    #    for i in range(len(grads)): grads[i] = torch.sum(grads[i], dim=0)
    if len(grads[0].shape) > 2:
        for i in range(len(grads)):
            if torch.is_tensor(grads[i]): grads[i] = grads[i].squeeze().numpy()
            grads[i] = viz._normalize_attr(grads[i], "absolute_value", reduction_axis=0)
            grads[i].squeeze()
            if isinstance(grads[i], np.ndarray):
                grads[i] = torch.from_numpy(grads[i])
    size = len(grads)
    data = np.empty(0)
    # print("Before:\n",data)

    for i in range(size):
        for j in range(i + 1, size):
            dcti = dct_2d(grads[i], norm='ortho')
            dctj = dct_2d(grads[j], norm='ortho')
            zigzagi = zigzag_scan(dcti)
            zigzagj = zigzag_scan(dctj)
            diff = torch.norm(dcti - dctj)
            #print(diff)
            data = np.append(data, diff.item())
            #print(data.size)
    # print("After:\n", data)
    return data

def generate_XB(compare_dict):
    grads = [compare_dict[i]["grad"] for i in range(len(compare_dict))]
    #if grads[0].shape[0] == 3:
    #    for i in range(len(grads)): grads[i] = torch.sum(grads[i], dim=0)
    if len(grads[0].shape) > 2:
        for i in range(len(grads)):
            if torch.is_tensor(grads[i]): grads[i] = grads[i].squeeze().numpy()
            grads[i] = viz._normalize_attr(grads[i], "absolute_value", reduction_axis=0)
            grads[i].squeeze()
            if isinstance(grads[i], np.ndarray):
                grads[i] = torch.from_numpy(grads[i])
    size = len(grads)
    data = np.empty(0)
    # print("Before:\n",data)

    for i in range(size//2):
        for j in range(size//2, size):
            dcti = dct_2d(grads[i], norm='ortho')
            dctj = dct_2d(grads[j], norm='ortho')
            zigzagi = zigzag_scan(dcti)
            zigzagj = zigzag_scan(dctj)
            diff = torch.norm(dcti - dctj)
            #print(diff)
            data = np.append(data, diff.item())
            #print(data.size)
    # print("After:\n", data)
    return data

def generate_compare_dict(algorithm, dataset, desired_classes, num_samples=20, is_XB=False):
    if algorithm == "iTAML":
       SalGenArgs.args = iTAMLArgs
    else: SalGenArgs.args = MnistArgs

    if dataset == "cifar100":
        SalGenArgs.class_per_task = 10
        SalGenArgs.num_class = 100
    else:
        SalGenArgs.class_per_task = 2
        SalGenArgs.num_class = 10

    SalGenArgs.algorithm = algorithm
    SalGenArgs.dataset = dataset
    SalGenArgs.args.dataset = SalGenArgs.dataset
    SalGenArgs.args.num_class = SalGenArgs.num_class


    model = load_model(algorithm, dataset, ses, args=SalGenArgs.args)
    sal_dataloader = sdl.SalDataloader(SalGenArgs)

    if is_XB:
        imgs1, _, _, STD, MEAN = sal_dataloader.load_data([desired_classes[0]], num_samples)
        imgs2, _, _, STD, MEAN = sal_dataloader.load_data([desired_classes[1]], num_samples)
        sal_imgs = torch.cat((imgs1,imgs2), dim=0)
    else:
        sal_imgs, _, _, STD, MEAN = sal_dataloader.load_data([desired_classes[0]], num_samples)
        #sal_imgs, _, _, STD, MEAN = sal_dataloader.load_data([desired_classes[1]], 20)

    # Reshape MNIST data for RPSnet
    if SalGenArgs.algorithm == "RPSnet" and SalGenArgs.dataset == "mnist":
        sal_imgs = sal_imgs.detach().numpy().reshape(-1, 784)
        sal_imgs = torch.from_numpy(sal_imgs)

    if algorithm == "RPSnet":
        infer_path = generate_path(ses, SalGenArgs.dataset, SalGenArgs.args)
        predicted = generate_predictions(algorithm, model, ses, sal_imgs, infer_path=infer_path)
    else:
        predicted = generate_predictions(algorithm, model, ses, sal_imgs, args=SalGenArgs.args)

    if SalGenArgs.algorithm == "foster" or SalGenArgs.algorithm == "memo" or SalGenArgs.algorithm == "der":
        saliency = Saliency(lambda x: model(x)["logits"])
    else:
        saliency = Saliency(model)

    compare_dict = {}
    num = 40 if is_XB else 20
    for ind in range(num):
        # for ind in range(len(desired_classes) * imgs_per_class):
        compare_dict[ind] = {"grad": None, "original": None, "pred": None}
        image = sal_imgs[ind].unsqueeze(0)
        image.requires_grad = True

        # Add additional arguments for RPSnet
        if SalGenArgs.algorithm == "RPSnet":
            grads = saliency.attribute(image, target=predicted[ind], abs=False,
                                       additional_forward_args=(infer_path, -1))
        else:
            grads = saliency.attribute(image, target=predicted[ind], abs=False)
        #grads = saliency.attribute(image, target=predicted[ind], abs=False)

        if SalGenArgs.dataset == "mnist":
            # Reshape MNIST data from RPSnet
            if SalGenArgs.algorithm == "RPSnet":
                grads = grads.reshape(28, 28)
            else:
                grads = grads.squeeze().cpu().detach()
            squeeze_grads = torch.unsqueeze(grads, 0)
            # Save gradient for comparison
            compare_dict[ind]["grad"] = grads
            grads = np.transpose(squeeze_grads.numpy(), (1, 2, 0))
        else:
            # Save gradient for comparison
            compare_dict[ind]["grad"] = grads
            grads = np.transpose(grads.squeeze().cpu().detach().numpy(), (1, 2, 0))

        # Reshape MNIST data from RPSnet
        if SalGenArgs.algorithm == "RPSnet" and SalGenArgs.dataset == "mnist":
            original_image = sal_imgs[ind].cpu().reshape(28, 28).unsqueeze(0)
        else:
            original_image = sal_imgs[ind].cpu()

        # Denormalization for RGB datasets
        if SalGenArgs.dataset != "mnist":
           original_image = original_image * STD[:, None, None] + MEAN[:, None, None]

        # Save image for comparison
        compare_dict[ind]["original"] = original_image
        original_image = np.transpose(original_image.detach().numpy(), (1, 2, 0))
    return compare_dict


if __name__ == '__main__':
    algorithm = "RPSnet"
    dataset = "cifar10"
    distill = True
    num = 10 if dataset == "cifar100" else 5
    num_imgs = 10

    #desired_classes = [0]
    #imgs_per_class = 10
    desired_classes = [0]
    imgs_per_class = 20
    is_XB = True

    rng = [0,9] if dataset == "cifar100" else [0,4]
    for ses in rng:
    #for ses in range(num):
        print(f'Session {ses}')
        print('#################################################################################')

        #if algorithm == "DGR" and distill:
        #    compare_dict = torch.load(f"SaliencyMaps/{algorithm}/{dataset}/distill/compare_dict_sess{ses}.pt")
        #else:
        #    compare_dict = torch.load(f"SaliencyMaps/{algorithm}/{dataset}/compare_dict_sess{ses}.pt")

        compare_dict = generate_compare_dict(algorithm, dataset, [0,1], imgs_per_class, is_XB)

        savepath = f"analysis/{algorithm}/{dataset}"
        if not os.path.exists(savepath):
            os.makedirs(savepath)

        if is_XB:
            savedata = generate_XB(compare_dict)
            savemat(f"analysis/{algorithm}/{dataset}/{algorithm}_ses_{ses}_XB_dcts.mat", {f"XB_{ses}_{algorithm}": savedata})
        else:
            savedata = generate_XA(compare_dict)
            savemat(f"analysis/{algorithm}/{dataset}/{algorithm}_ses_{ses}_XA_dcts.mat",
                    {f"XA_{ses}_{algorithm}": savedata})
            #savemat(f"matlab/{algorithm}/{dataset}/{algorithm}_ses_{ses}_XA1_dcts.mat", {f"XA_{ses}_{algorithm}": savedata})
        #if ses == 0: vmin, vmax = create_diff_heatmap(algorithm, dataset, ses, compare_dict)
        #else: create_diff_heatmap(algorithm, dataset, ses, compare_dict, vmin, vmax)
        print('\n\n')

        #input("Press 'Enter' to exit.")
