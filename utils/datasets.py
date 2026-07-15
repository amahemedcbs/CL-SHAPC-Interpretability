# Imports
from utils.model_parameters import iTAMLArgs


def get_dataset_params(dataset):
    # Normalize input string to lowercase to prevent minor spelling mismatches
    dataset_lower = dataset.lower()
    
    if dataset_lower == "cifar10":
        return SHAPCifar10
    if dataset_lower == "cifar100":
        return SHAPCifar100
    if dataset_lower == "tinyimagenet":
        return SHAPTinyImageNet
    if dataset_lower == "mnist":
        return SHAPMnist
    if dataset_lower == "pathmnist":
        return SHAPPathmnist
    if dataset_lower == "dermamnist":
        return SHAPDermamnist
    if dataset_lower == "octmnist":
        return SHAPOctmnist
    
    raise ValueError(f"Dataset '{dataset}' is not registered in get_dataset_params.")


class SHAPDataset:
    class_per_task = 2
    num_class = 10
    shap_samples = 100


class SHAPCifar10(SHAPDataset):
    class_per_task = 2
    num_class = 10
    num_task = 5
    shap_samples = 100


class SHAPCifar100(SHAPDataset):
    class_per_task = 10
    num_class = 100
    num_task = 10
    shap_samples = 20


class SHAPTinyImageNet(SHAPDataset):
    class_per_task = 20
    num_class = 200
    num_task = 10
    shap_samples = 20


class SHAPMnist(SHAPDataset):
    class_per_task = 2
    num_class = 10
    num_task = 5
    shap_samples = 50  # Set to a safe value for speedy local computation


class SHAPPathmnist(SHAPDataset):
    class_per_task = 3  # E.g., 9 classes split evenly across 3 task updates
    num_class = 9
    num_task = 3
    shap_samples = 50   # Number of samples per class evaluated by the explainer


class SHAPDermamnist(SHAPDataset):
    class_per_task = 2  # Adjust this depending on your PyCIL split setup
    num_class = 7
    num_task = 7
    shap_samples = 50


class SHAPOctmnist(SHAPDataset):
    class_per_task = 2  # E.g., 4 classes split across 2 task updates
    num_class = 4
    num_task = 2
    shap_samples = 50
