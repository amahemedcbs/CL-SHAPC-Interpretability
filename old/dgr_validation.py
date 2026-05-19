## SALIENCY MAP-MAKING

import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np

from image_crop import crop_and_combine_images, combine_cropped, combine_cropped_100

# Saliency Imports
from captum.attr import Saliency
from captum.attr import visualization as viz
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, Subset, DataLoader

# iTAML Imports
from basic_net import *


class iTAMLArgs:
    checkpoint = "results/cifar100/meta_mnist_T5_47"
    savepoint = "models/" + "/".join(checkpoint.split("/")[1:])
    data_path = "../Datasets/MNIST/"
    num_class = 10
    class_per_task = 2
    num_task = 5
    test_samples_per_class = 1000
    dataset = "mnist"
    optimizer = 'sgd'

    epochs = 20
    lr = 0.1
    train_batch = 256
    test_batch = 256
    workers = 16
    sess = 0
    schedule = [5, 10, 15]
    gamma = 0.5
    random_classes = False
    validation = 0
    memory = 2000
    mu = 1
    beta = 0.5
    r = 1


# RPSnet Imports
from rps_net import RPS_net_mlp, RPS_net_cifar, generate_path


class MnistArgs:
    epochs = 10
    checkpoint = "results/mnist/RPS_net_mnist"
    savepoint = "results/mnist/pathnet_mnist"
    dataset = "MNIST"
    num_class = 10
    class_per_task = 2
    M = 8
    L = 9
    N = 1
    lr = 0.001
    train_batch = 128
    test_batch = 128
    workers = 16
    resume = False
    arch = "res-18"
    start_epoch = 0
    evaluate = False
    sess = 0
    test_case = 0
    schedule = [6, 8, 16]
    gamma = 0.5
    rigidness_coff = 2.5
    jump = 1


class Cifar10Args:
    epochs = 10
    #    epochs = 2
    checkpoint = "results/cifar10/RPS_net_cifar10"
    savepoint = "results/cifar10/pathnet_cifar10"
    dataset = "CIFAR10"
    num_class = 10
    class_per_task = 2
    M = 8
    L = 9
    N = 1
    lr = 0.001
    train_batch = 128
    test_batch = 128
    workers = 16
    resume = False
    arch = "res-18"
    start_epoch = 0
    evaluate = False
    sess = 0
    test_case = 0
    schedule = [6, 8, 16]
    gamma = 0.5
    rigidness_coff = 2.5
    jump = 1


class Cifar100Args:
    checkpoint = "results/cifar100/RPS_CIFAR_M8_J1"
    labels_data = "prepare/cifar100_10.pkl"
    savepoint = ""

    num_class = 100
    class_per_task = 10
    M = 8
    jump = 2
    rigidness_coff = 2.5
    dataset = "CIFAR"

    epochs = 100
    epochs = 1
    L = 9
    N = 1
    lr = 0.001
    train_batch = 128
    test_batch = 128
    workers = 16
    resume = False
    arch = "res-18"
    start_epoch = 0
    evaluate = False
    sess = 0
    test_case = 0
    schedule = [20, 40, 60, 80]
    gamma = 0.5


# DGR Imports
import define_models as define
import dgr_parameters


class SalGenArgs:
    algorithm = "RPSnet"
    dataset = "mnist"
    args = None
    desired_classes = [0, 1]
    class_per_task = 2
    num_class = 10
    distill = False


device = "cuda:0" if torch.cuda.is_available() else "cpu"


def generate_predictions(algorithm, model, ses, images, **kwargs):
    if algorithm == "iTAML":
        model.set_saliency(True)
        outputs2 = model(images)
        pred = torch.argmax(outputs2[:, 0:SalGenArgs.class_per_task * (1 + ses)], 1, keepdim=False)
    elif algorithm == "RPSnet":
        outputs = model(images, kwargs['infer_path'], -1)
        _, pred = torch.max(outputs, 1)
    elif algorithm == "DGR":
        with torch.no_grad():
            scores = model.classify(images)
            _, pred = torch.max(scores, 1)
    predicted = pred.squeeze()

    return predicted


def get_indices(dataset, class_name):
    indices = []
    for i in range(len(dataset.targets)):
        for j in class_name:
            if dataset.targets[i] == j:
                indices.append(i)
    return indices


def load_saliency_data(dataset, desired_classes, imgs_per_class):
    transform = transforms.Compose(
        [transforms.ToTensor()])

    if not os.path.isdir(f"SaliencyMaps/{SalGenArgs.algorithm}/" + SalGenArgs.dataset):
        os.makedirs(f"SaliencyMaps/{SalGenArgs.algorithm}/" + SalGenArgs.dataset)

    saliencySet = datasets.MNIST(root=f"Datasets/{dataset}/", train=False,
                                 download=True,
                                 transform=transforms.Compose([transforms.ToTensor()]))
    MEAN = None
    STD = None


    idx = get_indices(saliencySet, desired_classes)

    subset = Subset(saliencySet, idx)

    # Create a DataLoader for the subset
    saliencyLoader = DataLoader(subset, batch_size=256)

    images, labels = next(iter(saliencyLoader))

    # Order saliency images and labels by class
    salIdx = []
    salLabels = []
    for i in range(len(desired_classes)):
        num = 0
        while len(salIdx) < imgs_per_class * (i + 1):
            if labels[num] == desired_classes[i]:
                salIdx.append(num)
                salLabels.append(desired_classes[i])
            #num += 4
            num += 16
    salImgs = images[salIdx]

    return salImgs, torch.tensor(salLabels), saliencySet.classes, MEAN, STD


def load_validation_data():
    import cv2
    import torch
    from torchvision.utils import make_grid

    labels = [2, 2, 2, 2, 2, 5, 5, 5, 5, 5]
    classes = ('0 - zero','1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine')

    images = []
    # Square
    #square = 255/2 * np.ones(shape=(28, 28, 1), dtype=np.float32)
    square = np.ones(shape=(28, 28, 1), dtype=np.float32)
    cv2.rectangle(square,
                  pt1=(0, 0),
                  pt2=(27, 27),
                  color=(255, 255, 255),
                  thickness=4)
    images.append(square)
    images.append(square)

    # X
    X = np.zeros(shape=(28, 28, 1), dtype=np.float32)
    cv2.line(X,
             pt1=(0, 0),
             pt2=(27, 27),
             color=(255, 255, 255),
             thickness=4)
    cv2.line(X,
             pt1=(27, 0),
             pt2=(0, 27),
             color=(255, 255, 255),
             thickness=4)
    images.append(X)

    # L
    L = np.zeros(shape=(28, 28, 1), dtype=np.float32)
    cv2.rectangle(L,
                  pt1=(0, 0),
                  pt2=(0, 27),
                  color=(255, 255, 255),
                  thickness=1)
    cv2.rectangle(L,
                  pt1=(0, 27),
                  pt2=(27, 27),
                  color=(255, 255, 255),
                  thickness=1)
    images.append(L)

    # Reverse L
    LReverse = np.zeros(shape=(28, 28, 1), dtype=np.float32)
    cv2.rectangle(LReverse,
                  pt1=(0, 27),
                  pt2=(27, 27),
                  color=(255, 255, 255),
                  thickness=1)
    cv2.rectangle(LReverse,
                  pt1=(27, 27),
                  pt2=(27, 0),
                  color=(255, 255, 255),
                  thickness=1)
    images.append(LReverse)

    # Corners
    corners = np.zeros(shape=(28, 28, 1), dtype=np.float32)
    cv2.rectangle(corners, pt1=(0, 0), pt2=(1, 1), color=(255, 255, 255), thickness=1)
    cv2.rectangle(corners, pt1=(0, 27), pt2=(1, 26), color=(255, 255, 255), thickness=1)
    cv2.rectangle(corners, pt1=(27, 0), pt2=(26, 1), color=(255, 255, 255), thickness=1)
    cv2.rectangle(corners, pt1=(27, 27), pt2=(26, 26), color=(255, 255, 255), thickness=1)
    images.append(corners)

    # Top
    top = np.zeros(shape=(28, 28, 1), dtype=np.float32)
    cv2.line(top,
             pt1=(0, 0),
             pt2=(27, 0),
             color=(255, 255, 255),
             thickness=1)
    images.append(top)

    # Left
    left = np.zeros(shape=(28, 28, 1), dtype=np.float32)
    cv2.line(left,
             pt1=(0, 0),
             pt2=(0, 27),
             color=(194, 194, 194),
             thickness=4)
    images.append(left)

    # Bottom
    bottom = np.zeros(shape=(28, 28, 1), dtype=np.float32)
    cv2.line(bottom,
             pt1=(0, 27),
             pt2=(27, 27),
             color=(125, 125, 125),
             thickness=4)
    images.append(bottom)

    # Right
    right = np.zeros(shape=(28, 28, 1), dtype=np.float32)
    cv2.line(right,
             pt1=(27, 0),
             pt2=(27, 27),
             color=(61, 61, 61),
             thickness=1)
    images.append(right)

    images = np.array(images)
    imgs_tensor = torch.from_numpy(images).permute(0, 3, 1, 2)
    grid = make_grid(imgs_tensor)
    grid_np = grid.permute(1, 2, 0).numpy()
    #plt.imshow(grid_np, cmap="gray")
    #plt.show()



    return imgs_tensor, torch.tensor(labels), classes, None, None


def create_saliency_map(model, ses, dataset, desired_classes, imgs_per_class):
    #sal_imgs, sal_labels, classes, MEAN, STD = load_saliency_data(dataset, desired_classes, imgs_per_class)
    sal_imgs, sal_labels, classes, MEAN, STD = load_validation_data()

    sal_imgs, sal_labels = sal_imgs.to(device), sal_labels.to(device)
    # sal_imgs.requires_grad_(True)

    # Add path argument for RPSnet
    predicted = generate_predictions(SalGenArgs.algorithm, model, ses, sal_imgs)
    print(predicted)

    saliency = Saliency(model)

    compare_grads = {}
    nrows, ncols = (2, 10) if SalGenArgs.dataset == "cifar100" else (2, 2 * imgs_per_class)
    fig, ax = plt.subplots(nrows, ncols, figsize=(15, 5))
    for ind in range(len(desired_classes)*imgs_per_class):
        compare_grads[ind] = {"grad": None, "original": None, "pred": None}
        img = sal_imgs[ind].unsqueeze(0)
        img.requires_grad_(True)
        #preds = model(input)

        grads = saliency.attribute(img, target=predicted[ind], abs=False)
        #grads = saliency.attribute(img, target=sal_labels[ind].item(), abs=False)
        #grads = torch.autograd.grad(preds[0][sal_labels[ind].item()], input)[0]

        # Reshape MNIST data from RPSnet
        grads = grads.squeeze().cpu().detach()
        squeeze_grads = torch.unsqueeze(grads, 0)
        # Save gradient for comparison
        compare_grads[ind]["grad"] = grads
        grads = np.transpose(squeeze_grads.numpy(), (1, 2, 0))


        truthStr = 'Truth: ' + str(classes[sal_labels[ind]])
        predStr = 'Pred: ' + str(classes[predicted[ind]])
        print(truthStr + '\n' + predStr)

        original_image = sal_imgs[ind].cpu()

        # Save image for comparison
        compare_grads[ind]["original"] = original_image
        original_image = np.transpose(original_image.detach().numpy(), (1, 2, 0))


        methods = ["original_image", "blended_heat_map"]
        signs = ["all", "absolute_value"]
        titles = ["Original Image", "Saliency Map"]
        colorbars = [False, True]

        # Check if image was misclassified
        if predicted[ind] != sal_labels[ind]:
            compare_grads[ind]["pred"] = False
            cmap = "Reds"
        else:
            compare_grads[ind]["pred"] = True
            cmap = "Blues"

        # Select row and column for saliency image
        if ind > imgs_per_class - 1:
            row, col = (1, ind - imgs_per_class)
        else:
            row, col = (0, ind)

        # Generate original images and saliency images
        for i in range(2):
            #print(f"Ind: {ind}\nRow: {row}\nCol: {col}\n")
            plt_fig_axis = (fig, ax[row][(2 * col) + i])
            _ = viz.visualize_image_attr(grads, original_image,
                                         method=methods[i],
                                         sign=signs[i],
                                         fig_size=(4, 4),
                                         plt_fig_axis=plt_fig_axis,
                                         cmap=cmap,
                                         show_colorbar=colorbars[i],
                                         title=titles[i],
                                         use_pyplot=False)
            if i == 0:
                ax[row][2 * col + i].images[0].set_cmap('gray')
                ax[row][2 * col + i].set_xlabel(truthStr)
            else:
                ax[row][2 * col + i].images[-1].colorbar.set_label(predStr)
            ax[row][2 * col + i].set_facecolor('gray')


    fig.tight_layout()
    if SalGenArgs.algorithm == "DGR" and SalGenArgs.distill:
        fig_save_path = f"DGRValidation/distill"
    else:
        fig_save_path = f"DGRValidation"
    fig.savefig(f"{fig_save_path}/Sess{ses}SalMap.png")
    plt.close()
    torch.save(compare_grads, f"{fig_save_path}/compare_dict_sess{ses}.pt")
    #fig.show()


def load_model(algorithm, dataset, ses, **kwargs):
    model = None
    model_path = ""
    match algorithm:
        case "iTAML":
            model_path = f"Saliency/{algorithm}/{dataset}/session_{ses}_model_best.pth.tar"
            model = BasicNet1(kwargs['args'], 0, device=device)
        case "RPSnet":
            model_path = f"Saliency/{algorithm}/{dataset}/session_{ses}_0_model_best.pth.tar"
            if dataset == "mnist":
                model = RPS_net_mlp(kwargs['args'])
            else:
                model = RPS_net_cifar(kwargs['args'])
        case "DGR":
            if SalGenArgs.distill:
                model_path = f"Saliency/{algorithm}/distill/model{ses + 1}"
            else:
                model_path = f"Saliency/{algorithm}/model{ses + 1}"
            model = define.define_classifier(args=dgr_parameters.args, config=dgr_parameters.config,
                                             device=dgr_parameters.device, depth=dgr_parameters.depth)

    model_data = torch.load(model_path, map_location=device, weights_only=False)

    print(model_data.keys())
    #print(model_data['state_dict'].keys())

    print(model.state_dict().keys())
    if algorithm == "RPSnet":
        model.load_state_dict(model_data['state_dict'])
    else:
        model.load_state_dict(model_data)
    model.eval()

    return model


def main(algorithm = None, dataset = None, start_sess = 0):

    if not algorithm or not dataset:
        algorithm = input("Which algorithm are you using? ")
        dataset = input("Which dataset are you using? ")

    SalGenArgs.algorithm = algorithm
    SalGenArgs.dataset = dataset

    num_sess = 5
    imgs_per_class = 5
    SalGenArgs.class_per_task = 2
    SalGenArgs.num_class = 10

    model = None
    for ses in range(start_sess, num_sess):
        if SalGenArgs.algorithm == "iTAML":
            SalGenArgs.args = iTAMLArgs
            SalGenArgs.args.dataset = SalGenArgs.dataset
            SalGenArgs.args.num_class = SalGenArgs.num_class
        else:
            if SalGenArgs.dataset == "mnist":
                SalGenArgs.args = MnistArgs
            elif SalGenArgs.dataset == "cifar10" or SalGenArgs.dataset == "svhn":
                SalGenArgs.args = Cifar10Args
            elif SalGenArgs.dataset == "cifar100":
                SalGenArgs.args = Cifar100Args

        model = load_model(SalGenArgs.algorithm, SalGenArgs.dataset, ses, args=SalGenArgs.args)

        print(f'Session {ses}')
        print('#################################################################################')
        create_saliency_map(model, ses, SalGenArgs.dataset, SalGenArgs.desired_classes, imgs_per_class)
        # create_saliency_map(model, ses, SalGenArgs.dataset, range(2), 5)
        print('\n\n')


if __name__ == '__main__':

    load_validation_data()
    #input("Press Enter to continue...")

    SalGenArgs.algorithm = "DGR"
    SalGenArgs.dataset = "mnist"
    SalGenArgs.distill = False

    params = ((0, [0,1]),
              (1, [2,3]),
              (2, [4,5]),
              (3, [6,7]),
              (4, [8,9]))

    end_sess = 5

    # '''
    #for k in range(end_sess):
    for k in range(1):

        start_sess, SalGenArgs.desired_classes = params[k]

        print(SalGenArgs.algorithm)
        main(SalGenArgs.algorithm, SalGenArgs.dataset, start_sess)

        if SalGenArgs.algorithm == "DGR" and SalGenArgs.distill:
            save_path = f"DGRValidation/distill"
        else:
            save_path = f"DGRValidation/"
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        imgs_per_sess = 2
        for j in range(imgs_per_sess):
            if SalGenArgs.algorithm == "DGR" and SalGenArgs.distill:
                image_path = f"DGRValidation/distill"
            else:
                image_path = f"DGRValidation"
            image_paths = [f'{image_path}/Sess{i}SalMap.png' for i in range(start_sess, end_sess)]
            crop_and_combine_images(image_paths,
                                    f"{save_path}/Class{SalGenArgs.desired_classes[j]}Cropped.png",
                                    False, #j+1)#(j*5)+1)
                                    (j*5)+4-j)

    # '''

    '''
    if SalGenArgs.algorithm == "DGR" and SalGenArgs.distill:
        crop_path = f"DGRValidation/distill"
        combined_path = f"DGRValidation/distill/All{SalGenArgs.dataset.capitalize()}Cropped.png"
    else:
        crop_path = f"DGRValidation"
        combined_path = f"DGRValidation/All{SalGenArgs.dataset.capitalize()}Cropped.png"
    cropped_paths = [f'{crop_path}/Class{i}Cropped.png' for i in range(10)]
    combine_cropped(cropped_paths, combined_path)
    '''

