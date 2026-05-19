from utils.datasets import get_dataset_params
from utils.model_parameters import get_algorithm_args

import os


shap_value_filepath_root = 'analysis/'

class SHAPArgs:
    def __init__(self, algorithm, dataset):
        self.algorithm = algorithm
        self.dataset_name = dataset
        self.dataset_params = get_dataset_params(dataset)
        self.algorithm_args = get_algorithm_args(self.algorithm, self.dataset_name)

def create_shap_value_filepath(shapArgs, first_last_only=True):
    total_samples = shapArgs.dataset_params.num_task * shapArgs.dataset_params.shap_samples * shapArgs.dataset_params.class_per_task

    if not os.path.isdir(f"{shap_value_filepath_root}{shapArgs.algorithm}/{shapArgs.dataset_name}"):
        os.makedirs(f"{shap_value_filepath_root}{shapArgs.algorithm}/{shapArgs.dataset_name}")

    if first_last_only:
        return f"{shap_value_filepath_root}{shapArgs.algorithm}/{shapArgs.dataset_name}/shap_values_first_last_{total_samples}"
    else:
        return f"{shap_value_filepath_root}{shapArgs.algorithm}/{shapArgs.dataset_name}/shap_values_full_{total_samples}"


def create_shapc_savepath(shapArgs, first_last_only=True, all_samples=False):
    if all_samples:
        total_samples = 1000
    else:
        total_samples = shapArgs.dataset_params.num_task * shapArgs.dataset_params.shap_samples * shapArgs.dataset_params.class_per_task

    if first_last_only:
        return f"{shap_value_filepath_root}{shapArgs.algorithm}/{shapArgs.dataset_name}/shapc_vals_first_last_{total_samples}.mat"
    else:
        return f"{shap_value_filepath_root}{shapArgs.algorithm}/{shapArgs.dataset_name}/shapc_vals_full_{total_samples}.mat"