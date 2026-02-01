import os

from calcSHAPC import *


def load_shapcs(path):
    if not os.path.isfile(savepath):
        scipy.io.savemat(savepath, {})
    mat_file = scipy.io.loadmat(savepath, simplify_cells=True)

    # Load the saved data, if possible
    keys_to_remove = ['__header__', '__version__', '__globals__']
    mat_dict = {key: value for key, value in mat_file.items() if key not in keys_to_remove}

    return mat_dict


if __name__ == "__main__":

    #algorithms = ["iTAML", "RPSnet", "DGR", "foster", "memo", "der", "icarl", "dsal", "tagfex", "xder"]
    algorithms = ["dsal"]
    dataset = "cifar100"

    if len(algorithms) > 1 and dataset != "mnist":
        algorithms.remove("DGR")

    for alg in algorithms:
        algorithm = alg
        num_sessions = 10 if dataset == "cifar100" else 5
        cls_per_task = 10 if dataset == "cifar100" else 2

        filepath = "shap_values_first_last_1000.npy"
        savepath = "shapc_vals_inclass"

        inclass_shapcs = load_shapcs(savepath)


        # Load the SHAP Values
        shap_values_loaded = np.load(f"analysis/{algorithm}/{dataset}/{filepath}", allow_pickle=True)  # ['shap_dict']
        num_imgs = len(shap_values_loaded[()].keys())
        shap_dict = {}
        for i in range(num_imgs):
            shap_dict[f'{i}'] = shap_values_loaded[()][f'{i}']

        for cls in range((num_sessions-1)*cls_per_task):
            shapc_dict = {}
            for sample in tqdm(range(num_imgs), desc="Progress"):
                # If computing inclass shapc and the sample is from the wrong class
                if shap_dict[f'{sample}']['true_label'] != cls: continue

                start_sess = shap_dict[f'{sample}']['true_label'] // cls_per_task
                # If the sample is from the last task skip
                if start_sess >= num_sessions-1: continue

                range1 = [start_sess]

                for ses in range1:
                    #print("Ses:", ses)
                    #print("Sample:", sample)
                    shap_value1 = shap_dict[f'{sample}'][f'ses{ses}']['shap_values']
                    shap_value1 = normalize_shap_value(shap_value1)

                    if algorithm == "iTAML": range2 = [num_sessions - 1 + ses]
                    else: range2 = [num_sessions-1]

                    for j in range2:

                        shap_value2 = shap_dict[f'{sample}'][f'ses{j}']['shap_values']
                        shap_value2 = normalize_shap_value(shap_value2)

                        p_tau = get_important_feature_area(shap_value1, percentile=70)
                        p_t = get_important_feature_area(shap_value2, percentile=70)

                        # --- Step 3: Calculate SHAPC ---
                        shapc_value = calculate_shapc(shap_value1, p_tau, shap_value2, p_t)
                        last_task = j-ses if algorithm == "iTAML" else j
                        if f'sc{ses}{last_task}' not in shapc_dict: shapc_dict[f'sc{ses}{last_task}'] = {}
                        shapc_dict[f'sc{ses}{last_task}'][f'sample{sample}'] = shapc_value
            inclass_shapcs[f'cls{cls}'] = shapc_dict
        save_path = f"analysis/{algorithm}/{dataset}/{savepath}.mat"
        scipy.io.savemat(save_path, inclass_shapcs)
        pass