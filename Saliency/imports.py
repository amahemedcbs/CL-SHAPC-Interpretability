# iTAML Imports
from basic_net import *


# RPSnet Imports
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


# DGRv2 Imports
from Saliency.DGR.models import CNN, WGAN
from Saliency.DGR.dgr import Scholar
#import copy, math, random

class SalGenArgs:
    algorithm = "RPSnet"
    dataset = "mnist"
    args = None
    desired_classes = [0, 1]
    class_per_task = 2
    num_class = 10
    distill = False

# PyCIL Imports
pycil_algs = ["der", "foster", "memo", "icarl", "simplecil", "ds-al"]
from Saliency.PyCIL.models import foster
from Saliency.PyCIL.utils import factory
from Saliency.PyCIL.utils.inc_net import FOSTERNet, AdaptiveNet, DERNet, IncrementalNet, SimpleCosineIncrementalNet
from Saliency.PyCIL.utils.inc_net import DSALNet, ACILNet, BaseNet

fosterArgs = {'config': './exps/fostertest.json', 'prefix': 'cil', 'dataset': 'cifar100',
              'memory_size': 2000, 'memory_per_class': 20, 'fixed_memory': True,
              'shuffle': False, 'init_cls': 10, 'increment': 10, 'model_name': 'foster',
              'convnet_type': 'resnet32', 'device': ['0'], 'seed': [1993], 'beta1': 0.96,
              'beta2': 0.97, 'oofc': 'ft', 'is_teacher_wa': False, 'is_student_wa': False,
              'lambda_okd': 1, 'wa_value': 1, 'init_epochs': 1, 'init_lr': 0.1,
              'init_weight_decay': 0.0005, 'boosting_epochs': 1, 'compression_epochs': 1,
              'lr': 0.1, 'batch_size': 128, 'weight_decay': 0.0005, 'num_workers': 8, 'T': 2}

memoArgs = {"prefix": "benchmark", "dataset": "cifar100", "memory_size": 2000,
            "memory_per_class":20, "fixed_memory": False, "shuffle": False, "init_cls": 10,
            "increment": 10, "model_name": "memo", "convnet_type": "memo_resnet32",
            "train_base": True, "train_adaptive": False, "debug": False, "skip": False,
            "device": ["0"], "seed":[1993], "scheduler": "steplr", "init_epoch": 200,
            "t_max": None, "init_lr" : 0.1, "init_weight_decay" : 5e-4, "init_lr_decay" : 0.1,
            "init_milestones" : [60,120,170], "milestones" : [80,120,150], "epochs": 170,
            "lrate" : 0.1, "batch_size" : 128, "weight_decay" : 2e-4, "lrate_decay" : 0.1,
            "alpha_aux" : 1.0}

derArgs = {"prefix": "reproduce", "dataset": "cifar10", "memory_size": 2000,
           "memory_per_class": 20, "fixed_memory": False, "shuffle": False, "init_cls": 2,
           "increment": 2, "model_name": "der", "convnet_type": "resnet32",
           "device": ["0"], "seed": [1993]}

icarlArgs = {"prefix": "reproduce", "dataset": "cifar10", "memory_size": 2000,
             "memory_per_class": 20, "fixed_memory": False, "shuffle": False, "init_cls": 2,
             "increment": 2, "model_name": "icarl", "convnet_type": "resnet32",
             "device": ["0"], "seed": [1993]}

scilArgs = {"prefix": "reproduce", "dataset": "cifar10", "memory_size": 0, "memory_per_class": 0,
            "fixed_memory": False, "shuffle": False, "init_cls": 50, "increment": 10,
            "model_name": "simplecil", "convnet_type": "cosine_resnet32", "device": ["0"],
            "seed": [1993], "init_epoch": 200, "init_lr": 0.01, "batch_size": 128,
            "weight_decay": 0.05, "init_lr_decay": 0.1, "init_weight_decay": 5e-4,
            "min_lr": 0}

dsalArgs = {"model_name": "ds-al", "prefix": "DS-AL", "memory_size": 0, "dataset": "cifar10",
            "seed": [1993], "shuffle": False, "device": ["0"], "convnet_type": "resnet32",
            "init_cls": 2, "increment": 2, "num_workers": 16, "init_batch_size": 128,
            "IL_batch_size": 4096, "inplace_repeat": 1,

            "configurations": {
                "cifar10": {
                    "buffer_size": 4096,
                    "gamma": 0.1,
                    "gamma_comp": 0.1,
                    "compensation_ratio": 0.6,
                    "init_weight_decay": 5e-4,
                    "scheduler": {
                        "type": "MultiStep",
                        "init_lr": 0.1,
                        "init_epochs": 160,
                        "warmup": 0,
                        "milestones": [120, 140],
                        "decay": 0.1
                    }
                },
                "cifar100": {
                    "buffer_size": 8192,
                    "gamma": 0.1,
                    "gamma_comp": 0.1,
                    "compensation_ratio": 0.6,
                    "init_weight_decay": 5e-4,
                    "scheduler": {
                        "type": "MultiStep",
                        "init_lr": 0.1,
                        "init_epochs": 160,
                        "warmup": 0,
                        "milestones": [120, 140],
                        "decay": 0.1
                    }
                }
            }
        }
