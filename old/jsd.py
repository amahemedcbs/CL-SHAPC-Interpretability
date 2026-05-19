import copy
import os.path
import sys

import numpy as np
from torch_dct import dct_2d
from feature_comparison import zigzag_scan
from scipy.stats import gaussian_kde
from scipy.spatial.distance import jensenshannon
import scipy.io
import matplotlib.pyplot as plt
from tqdm import tqdm


# Saliency Imports
import torch
from captum.attr import Saliency
from captum.attr import visualization as viz

from saliency_generator import load_model, generate_predictions
from saliency_generator import SalGenArgs, iTAMLArgs, MnistArgs
import saliency_dataloader as sdl
from rps_net import generate_path



#device = 'xpu' if torch.xpu.is_available() else 'cpu'
#print(f"Using the {device}...")


def setup_args(algorithm, dataset) -> None:
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


def randomly_select_images(algorithm, dataset, desired_class, num_samples=20):
    sal_dataloader = sdl.SalDataloader(SalGenArgs)

    sal_imgs, _, _, STD, MEAN = sal_dataloader.load_data([desired_class], num_samples, shuffle=True)
    #sal_imgs = sal_imgs.to(device)

    # Reshape MNIST data for RPSnet
    if SalGenArgs.algorithm == "RPSnet" and SalGenArgs.dataset == "mnist":
        sal_imgs = sal_imgs.detach().numpy().reshape(-1, 784)
        sal_imgs = torch.from_numpy(sal_imgs)
    return sal_imgs, STD, MEAN


def create_compare_dict(algorithm, dataset, cls, imgs, STD=None, MEAN=None):
    # Load models
    sessions = [cls//2]
    sessions.append(9) if dataset == "cifar100" else sessions.append(4)
    #print(sessions)
    initial_model = load_model(algorithm, dataset, sessions[0], args=SalGenArgs.args)
    final_model = load_model(algorithm, dataset, sessions[1], args=SalGenArgs.args)
    models_sessions = [(initial_model, sessions[0]), (final_model, sessions[1])]
    compare_dicts = []


    for model, ses in models_sessions:
        # For RPSnet
        if algorithm == "RPSnet":
            infer_path = generate_path(ses, SalGenArgs.dataset, SalGenArgs.args)
            predicted = generate_predictions(algorithm, model, ses, imgs, infer_path=infer_path)
        else:
            predicted = generate_predictions(algorithm, model, ses, imgs, args=SalGenArgs.args)

        if SalGenArgs.algorithm == "foster" or SalGenArgs.algorithm == "memo" or SalGenArgs.algorithm == "der":
            saliency = Saliency(lambda x: model(x)["logits"])
        else:
            saliency = Saliency(model)

        compare_dict = {}
        num_samples = len(imgs)
        for ind in range(num_samples):
            # for ind in range(len(desired_classes) * imgs_per_class):
            #compare_dict[ind] = {"grad": None, "original": None, "pred": None}
            compare_dict[ind] = {"grad": None, "original": None}
            image = imgs[ind].unsqueeze(0)
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
                original_image = imgs[ind].cpu().reshape(28, 28).unsqueeze(0)
            else:
                original_image = imgs[ind].cpu()

            # Denormalization for RGB datasets
            if SalGenArgs.dataset != "mnist":
               original_image = original_image * STD[:, None, None] + MEAN[:, None, None]

            # Save image for comparison
            compare_dict[ind]["original"] = original_image
            original_image = np.transpose(original_image.detach().numpy(), (1, 2, 0))
        compare_dicts.append(compare_dict)
    return compare_dicts


def create_out_of_class_dict(cls0dict, cls1dict) -> dict:
    combined_dict = {}

    map_num = 0
    for key in cls0dict:
        combined_dict[map_num] = cls0dict[key]
        map_num += 1
    for key in cls1dict:
        combined_dict[map_num] = cls1dict[key]
        map_num += 1

    #print("Finished creating dict.\nMap Num:", map_num)
    return combined_dict


def convert_compare_dict_for_saving(cdict):

    import copy

    cdict = copy.deepcopy(cdict)
    string_dicts = []
    for d in range(len(cdict)):
        c = {}
        for key, value in cdict[d].items():
            if isinstance(key, int) or isinstance(key, float):
                c[f"map{key}"] = value
            else:
                c[key] = value
        string_dicts.append(c)

    '''
    converted_dicts = copy.deepcopy(string_dicts)
    for ind in range(len(string_dicts)):
        for img in string_dicts[ind].keys():
            for key in string_dicts[ind][img].keys():
                converted_dicts[ind][img][key] = string_dicts[ind][img][key].numpy()
                
    return converted_dicts
    #'''
    return string_dicts


def convert_compare_dict_for_loading(cdict):

    import copy

    cdict = copy.deepcopy(cdict)
    string_dicts = []
    for d in range(len(cdict)):
        c = {}
        for key, value in cdict[d].items():
            if "map" in key:
                c[int(key.replace("map", ""))] = value
            else:
                c[key] = value
        string_dicts.append(c)

    return string_dicts


def generate_XA(compare_dict) -> np.ndarray:
    grads = []
    for i in range(len(compare_dict)):
        grad = compare_dict[i]["grad"]
        #'''
        if torch.is_tensor(grad): grad = grad.squeeze().numpy()
        if len(grad.shape) > 2:
            grad = viz._normalize_attr(grad, "absolute_value", reduction_axis=0)
        #'''
        if isinstance(grad, np.ndarray):
            grad = torch.from_numpy(grad)
        #grad = grad.to(device)
        grads.append(grad.squeeze())
    #grads = [torch.from_numpy(compare_dict[i]["grad"]).squeeze() for i in range(len(compare_dict))]
    #if grads[0].shape[0] == 3:
    #    for i in range(len(grads)): grads[i] = torch.sum(grads[i], dim=0)
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


def generate_XB(compare_dict) -> np.ndarray:
    grads = []
    for i in range(len(compare_dict)):
        grad = compare_dict[i]["grad"]
        #'''
        if torch.is_tensor(grad): grad = grad.squeeze().numpy()
        if len(grad.shape) > 2:
            grad = viz._normalize_attr(grad, "absolute_value", reduction_axis=0)
        #'''
        if isinstance(grad, np.ndarray):
            grad = torch.from_numpy(grad)
        # grad = grad.to(device)
        grads.append(grad.squeeze())
    #grads = [torch.from_numpy(compare_dict[i]["grad"]).squeeze() for i in range(len(compare_dict))]
    #if grads[0].shape[0] == 3:
    #    for i in range(len(grads)): grads[i] = torch.sum(grads[i], dim=0)
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


def generate_maps(algorithm, dataset, desired_classes, num_samples=60, num_trials=50):

    setup_args(algorithm, dataset)

    if dataset == "cifar100":
        paths = ['_ses_0_XA_dcts.mat', '_ses_9_XA_dcts.mat']
    else:
        paths = ['_ses_0_XA_dcts.mat', '_ses_4_XA_dcts.mat']

    if desired_classes == 'all':
        desired_classes = range(10)

    mat_dict = {}
    for i in range(1, num_trials + 1):
        for cls in desired_classes:
            if f"t{i}" not in mat_dict.keys(): mat_dict[f"t{i}"] = {}
            mat_dict[f"t{i}"][f"c{cls}_maps"] = {"ses0": {}, "ses4": {}}

    for trial in tqdm(range(1, num_trials + 1), desc="Progress"):

        # At the beginning of every trial, randomly select images and generate saliency maps
        for cls in desired_classes:
            # Select images
            imgs, STD, MEAN = randomly_select_images(algorithm, dataset, cls, num_samples)
            # Generate maps (compare dict)
            compare_dicts = create_compare_dict(algorithm, dataset, cls, imgs, STD, MEAN)
            compare_dicts_converted = convert_compare_dict_for_saving(compare_dicts)
            mat_dict[f"t{trial}"][f"c{cls}_maps"]["ses0"] = compare_dicts_converted[0]
            mat_dict[f"t{trial}"][f"c{cls}_maps"]["ses4"] = compare_dicts_converted[1]

    save_path = f"analysis/{algorithm}/{dataset}/saved_maps_alt.mat"
    scipy.io.savemat(save_path, mat_dict)


def calc_jsd(dcts1, dcts2):

    # Generate x points for PDF
    combined_data = np.concatenate((dcts1, dcts2))
    xmin = np.min(combined_data) - 3 * np.std(combined_data)
    xmax = np.max(combined_data) + 3 * np.std(combined_data)
    num_points = 500
    x_common = np.linspace(xmin, xmax, num_points)

    # Generate PDFs
    kde1 = gaussian_kde(dcts1)
    pdf1 = kde1(x_common)
    kde2 = gaussian_kde(dcts2)
    pdf2 = kde2(x_common)

    js_distance = jensenshannon(pdf1, pdf2)
    # The Jensen-Shannon Divergence is the square of the distance
    js_divergence = js_distance**2

    return js_distance


def jsd_protocol(algorithm, dataset, desired_classes, saved_data, num_samples=60, num_trials=50, scenarios=None):

    if dataset == "cifar100":
        paths = ['_ses_0_XA_dcts.mat', '_ses_9_XA_dcts.mat']
    else:
        if scenarios == [9]: paths = ['_ses_3_XA_dcts.mat','_ses_4_XA_dcts.mat']
        else: paths = ['_ses_0_XA_dcts.mat','_ses_4_XA_dcts.mat']

    if scenarios is None:
        scenarios = [1,2]
    '''
    This is the format of the dictionary for the mat file
    
    mat_dict = {"t1":
                        {"c0_maps": {"ses0":{}, "ses4":{}},
                         "c1_maps": {"ses0":{}, "ses4":{}},
                         "scn1": 0.0, "scn2": 0.0, "scn3": 0.0,
                         "scn4": 0.0}
                    }
    '''

    # Load the saved data, if possible
    keys_to_remove = ['__header__', '__version__', '__globals__']
    mat_dict = {key: value for key, value in saved_data.items() if key not in keys_to_remove}

    if mat_dict == {}:
        for i in range(1, num_trials + 1):
            mat_dict[f"t{i}"] = {
                "c0_maps": {"ses0": {}, "ses4": {}},
                "c1_maps": {"ses0": {}, "ses4": {}},
                "scn1": 0.0, "scn2": 0.0, "scn3": 0.0,
                "scn4": 0.0, "scn9": 0.0
            }

    for scn in scenarios:

        distances = []

        for trial in tqdm(range(1, num_trials + 1), desc=f"Scenario {scn} Progress"):

            # At the beginning of every trial, randomly select images and generate saliency maps
            if mat_dict[f"t{trial}"]["c0_maps"] == {"ses0":{}, "ses4":{}}:
                # Select images
                cls0_imgs, STD, MEAN = randomly_select_images(algorithm, dataset, desired_classes[0], num_samples)
                # Generate maps (compare dict)
                cls0_compare_dicts = create_compare_dict(algorithm, dataset, desired_classes[0], cls0_imgs, STD, MEAN)
                cls0_compare_dicts_converted = convert_compare_dict_for_saving(cls0_compare_dicts)
                mat_dict[f"t{trial}"]["c0_maps"]["ses0"] = cls0_compare_dicts_converted[0]
                mat_dict[f"t{trial}"]["c0_maps"]["ses4"] = cls0_compare_dicts_converted[1]
            else:
                cls0_compare_dicts = [mat_dict[f"t{trial}"]["c0_maps"]["ses0"], mat_dict[f"t{trial}"]["c0_maps"]["ses4"]]
                cls0_compare_dicts = convert_compare_dict_for_loading(cls0_compare_dicts)


            if mat_dict[f"t{trial}"]["c1_maps"] == {"ses0":{}, "ses4":{}}:
                cls1_imgs, STD, MEAN = randomly_select_images(algorithm, dataset, desired_classes[1], num_samples)
                cls1_compare_dicts = create_compare_dict(algorithm, dataset, desired_classes[0], cls1_imgs, STD, MEAN)
                cls1_compare_dicts_converted = convert_compare_dict_for_saving(cls1_compare_dicts)
                mat_dict[f"t{trial}"]["c1_maps"]["ses0"] = cls1_compare_dicts_converted[0]
                mat_dict[f"t{trial}"]["c1_maps"]["ses4"] = cls1_compare_dicts_converted[1]
            else:
                cls1_compare_dicts = [mat_dict[f"t{trial}"]["c1_maps"]["ses0"], mat_dict[f"t{trial}"]["c1_maps"]["ses4"]]
                cls1_compare_dicts = convert_compare_dict_for_loading(cls1_compare_dicts)


            match scn:
                case 1 | 9:
                    initial_dcts = generate_XA(cls0_compare_dicts[0]) # Create XA_0
                    final_dcts = generate_XA(cls0_compare_dicts[1])  # Create XA_4 (or XA_9)
                case 2:
                    initial_dcts = generate_XA(cls1_compare_dicts[0])  # Create XA_0
                    final_dcts = generate_XA(cls1_compare_dicts[1])  # Create XA_4 (or XA_9)
                case 3:
                    initial_dcts = generate_XA(cls0_compare_dicts[0])  # Create XA_0 for cls 0, ses 0

                    out_of_class_dict = create_out_of_class_dict(cls0_compare_dicts[0], cls1_compare_dicts[0])
                    final_dcts = generate_XB(out_of_class_dict)  # Create XB_0 for cls 0 vs cls 1, ses 0
                case 4:
                    initial_dcts = generate_XA(cls0_compare_dicts[1])  # Create XA_0 for cls 0, ses 4

                    out_of_class_dict = create_out_of_class_dict(cls0_compare_dicts[1], cls1_compare_dicts[1])
                    final_dcts = generate_XB(out_of_class_dict)  # Create XB_0 for cls 0 vs cls 1, ses 4

            distance = calc_jsd(initial_dcts, final_dcts)
            mat_dict[f"t{trial}"][f"scn{scn}"] = distance

    return mat_dict


def fullJSD(algorithm, dataset, printpar=False, plot=False):

    if dataset == "cifar100":
        paths = [['_ses_0_XA_dcts.mat', '_ses_9_XA_dcts.mat'],
                 ['_ses_0_XB_dcts.mat', '_ses_9_XB_dcts.mat'],
                 ['_ses_0_XA_dcts.mat', '_ses_0_XB_dcts.mat'],
                 ['_ses_9_XA_dcts.mat', '_ses_9_XB_dcts.mat']]

        titles = ["XA_0 vs XA_9 (Hyp = Low)", "XB_0 vs XB_9 (Hyp = Low)",
                  "XA_0 vs XB_0 (Hyp = High)", "XA_9 vs XB_9 (Hyp = High)"]
    else:
        paths = [['_ses_0_XA_dcts.mat','_ses_4_XA_dcts.mat'],
                 ['_ses_0_XB_dcts.mat','_ses_4_XB_dcts.mat'],
                 ['_ses_0_XA_dcts.mat','_ses_0_XB_dcts.mat'],
                 ['_ses_4_XA_dcts.mat','_ses_4_XB_dcts.mat']]

        titles = ["XA_0 vs XA_4 (Hyp = Low)", "XB_0 vs XB_4 (Hyp = Low)",
                  "XA_0 vs XB_0 (Hyp = High)", "XA_4 vs XB_4 (Hyp = High)"]

    distances = []

    for i in range(len(paths)):
        class1_data = scipy.io.loadmat(f'matlab/{algorithm}/{dataset}/{algorithm}{paths[i][0]}')
        fns1 = list(class1_data.keys())
        data1 = class1_data[fns1[-1]].flatten()

        class2_data = scipy.io.loadmat(f'matlab/{algorithm}/{dataset}/{algorithm}{paths[i][1]}')
        fns2 = list(class2_data.keys())
        data2 = class2_data[fns2[-1]].flatten()


        # Generate x points for PDF
        combined_data = np.concatenate((data1, data2))
        xmin = np.min(combined_data) - 3 * np.std(combined_data)
        xmax = np.max(combined_data) + 3 * np.std(combined_data)
        num_points = 500
        x_common = np.linspace(xmin, xmax, num_points)

        # Generate PDFs
        kde1 = gaussian_kde(data1)
        pdf1 = kde1(x_common)
        kde2 = gaussian_kde(data2)
        pdf2 = kde2(x_common)

        js_distance = jensenshannon(pdf1, pdf2)
        # The Jensen-Shannon Divergence is the square of the distance
        js_divergence = js_distance**2

        if printpar:
            if i == 0:
                print("Jensen-Shannon Distances (Metric)")
                print("----------------------------------")
            print(f"{titles[i]}: {js_distance:.4f}")
            #print("Jensen-Shannon Divergence:", js_divergence)

        if plot:
            # 5. Optional: Plot the estimated PDFs
            plt.figure(figsize=(10, 6))
            plt.plot(x_common, pdf1, label='PDF 1')
            plt.plot(x_common, pdf2, label='PDF 2')
            plt.xlabel('Value')
            plt.ylabel('Probability Density')
            plt.title('Estimated Probability Density Functions')
            plt.legend()
            plt.grid(True)
            plt.show()

        distances.append(js_distance)

    return distances[0], distances[1], distances[2], distances[3]


def create_jsd_table(dataset):
    from tabulate import tabulate

    if dataset == "cifar100":
        headers = ["Algorithm", "XA_0 vs XA_9", "XA_0 vs XB_0", "XA_9 vs XB_9"]
    else:
        headers = ["Algorithm", "XA_0 vs XA_4", "XA_0 vs XB_0", "XA_4 vs XB_4"]

    algs = ["iTAML", "RPSnet", "dgr", "foster", "memo", "der"]
    if dataset != "mnist":
        algs.remove("dgr")
    jsdTable = []

    for alg in algs:
        dist1, dist2, dist3, dist4 = fullJSD(alg, dataset)
        entry = [alg, dist1.round(4), dist3.round(4), dist4.round(4)]
        jsdTable.append(entry)


    print(tabulate(jsdTable, headers=headers, tablefmt="fancy_grid"))

if __name__ == '__main__':
    algorithm = "iTAML"
    dataset = "cifar10"

    '''
    algorithm = sys.argv[1]
    dataset = sys.argv[2]
    #'''
    desired_classes = [4, 5]
    #desired_classes = 'all'
    scenarios = [1,2]

    generate_maps(algorithm, dataset, desired_classes)

    '''
    if scenarios == [9]:
        save_path = f"analysis/{algorithm}/{dataset}/inclass_jsd_alt.mat"
    else:
        save_path = f"analysis/{algorithm}/{dataset}/inclass_jsd.mat"
    #save_path = f"matlab/{algorithm}/{dataset}/test_jsd.mat"
    if not os.path.isfile(save_path):
        scipy.io.savemat(save_path, {})
    saved_data = scipy.io.loadmat(save_path, simplify_cells=True)
    #jsd(algorithm, dataset)
    #create_jsd_table(dataset)
    setup_args(algorithm, dataset)
    results = jsd_protocol(algorithm, dataset, desired_classes, saved_data,  scenarios=scenarios)
    scipy.io.savemat(save_path, results)
    #'''
    #fullJSD(algorithm, dataset, printpar=True)
    pass























# 3. Handle potential negative values
#pdf1[pdf1 < 0] = 0
#pdf2[pdf2 < 0] = 0