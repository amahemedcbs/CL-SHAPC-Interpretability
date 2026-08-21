import torch
import argparse
import os
import torch.optim as optim
import pickle
import numpy as np
import copy
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

from models.iTAML.resnet import *
from models.iTAML.radam import *

from models.iTAML import incremental_dataloader as data
from utils.model_parameters import iTAMLArgs

from utils.model_parameters import pycil_algs, get_algorithm_args
from utils.model_parameters import FOSTERNet, AdaptiveNet, DERNet, IncrementalNet, DSALNet, TagFexNet
from utils.model_parameters import BasicNet1, RPS_net_cifar, RPS_net_mlp
from utils.model_parameters import XDer, mammoth_load_checkpoint, get_backbone_class, get_dataset_class, original_cwd

device = "cuda:0" if torch.cuda.is_available() else "cpu"



def compute_accuracy(predictions, targets):
    correct, total = 0, 0
    correct += (predictions.cpu() == targets).sum()
    total += len(targets)
    correct = correct.cpu().data.numpy()
    return np.around(correct * 100 / total, decimals=2)


def generate_predictions(algorithm, model, ses, images, labels=None, **kwargs):
    if not isinstance(images, torch.Tensor):
        images = torch.tensor(images)
    images = images.to(device)

    if images.ndim == 3:
        images = images.unsqueeze(0)

    if algorithm == "iTAML":
        model.set_shap(True)
        outputs2 = model(images)
        pred = torch.argmax(outputs2[:, 0:kwargs['cls_per_task'] * (1 + ses)], dim=-1)
    elif algorithm == "RPSnet":
        outputs = model(images, kwargs['infer_path'], -1)
        pred = torch.argmax(outputs, dim=-1)
    elif algorithm == "DGR":
        real_scores = model.forward(images)
        pred = torch.argmax(real_scores[:, 0: kwargs['cls_per_task'] * (1 + ses)], dim=-1)
    elif algorithm == "xder":
        outputs = model(images)
        pred = torch.argmax(outputs[:, :kwargs['cls_per_task'] * (1 + ses)].data, dim=-1)
    else:
        # Fast, pooled PyCIL prediction (DS-AL, DER, FOSTER, MEMO, TagFEx, iCaRL)
        with torch.no_grad():
            if hasattr(model, 'convnets') and len(model.convnets) > 0:
                feats = []
                for conv in model.convnets:
                    f = conv(images)
                    if isinstance(f, dict):
                        f = f.get("features", list(f.values())[0])
                    if isinstance(f, torch.Tensor) and f.ndim == 4:
                        f = F.adaptive_avg_pool2d(f, (1, 1))
                    f = torch.flatten(f, 1)
                    feats.append(f)
                feat = torch.cat(feats, dim=1)

                if hasattr(model, 'fc') and isinstance(model.fc, nn.Linear):
                    if feat.shape[1] == model.fc.in_features:
                        logits = model.fc(feat)
                    elif feat.shape[1] < model.fc.in_features:
                        pad = torch.zeros((feat.shape[0], model.fc.in_features - feat.shape[1]), device=feat.device)
                        logits = model.fc(torch.cat([feat, pad], dim=1))
                    else:
                        logits = model.fc(feat[:, :model.fc.in_features])
                else:
                    logits = feat
            else:
                out = model(images)
                if isinstance(out, dict):
                    logits = out.get("logits", out.get("features", list(out.values())[0]))
                else:
                    logits = out
                if isinstance(logits, torch.Tensor) and logits.ndim == 4:
                    logits = F.adaptive_avg_pool2d(logits, (1, 1)).flatten(1)
                if hasattr(model, 'fc') and isinstance(model.fc, nn.Linear) and logits.shape[-1] == model.fc.in_features:
                    logits = model.fc(logits)

            # Restrict logits to valid classification outputs
            if hasattr(model, 'fc') and isinstance(model.fc, nn.Linear) and logits.shape[-1] > model.fc.out_features:
                logits = logits[:, :model.fc.out_features]
            elif logits.shape[-1] > 7:
                logits = logits[:, :7]

            pred = torch.argmax(logits, dim=-1)

    predicted = pred.squeeze()
    if labels is not None:
        acc = compute_accuracy(predicted, labels)
        print("Acc:", acc)

    return predicted


def _extract_convnet(convnet_obj):
    """Helper to safely extract the nn.Module from get_convnet return value"""
    if isinstance(convnet_obj, tuple):
        return convnet_obj[0]
    return convnet_obj


def load_model(algorithm, dataset, session, shapArgs=None, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Support both hyphen and non-hyphen naming formats (ds-al vs dsal)
    algs_to_try = [algorithm]
    if algorithm == "dsal":
        algs_to_try.append("ds-al")
    elif algorithm == "ds-al":
        algs_to_try.append("dsal")

    possible_paths = []
    for alg in algs_to_try:
        possible_paths.extend([
            f"savedmodels/{alg}/{dataset}/{alg}_{dataset}_{session}.pth",
            f"savedmodels/{alg}/{dataset}/{alg}_ses_{session}.pkl",
            f"savedmodels/{alg}/{dataset}/{alg}_{dataset}_{session}.pkl",
            f"savedmodels/{alg}/{dataset}/{alg}_{session}.pkl",
            f"/content/drive/MyDrive/CL-SHAPC-Interpretability/savedmodels/{alg}/{dataset}/{alg}_{dataset}_{session}.pth",
            f"/content/drive/MyDrive/CL-SHAPC-Interpretability/savedmodels/{alg}/{dataset}/{alg}_ses_{session}.pkl",
            f"/content/drive/MyDrive/CL-SHAPC-Interpretability/savedmodels/{alg}/{dataset}/{alg}_{dataset}_{session}.pkl",
            f"/content/drive/MyDrive/CL-SHAPC-Interpretability/savedmodels/{alg}/{dataset}/{alg}_{session}.pkl",
            f"checkpoints/{alg}_{dataset}_{session}.pkl"
        ])
    
    model_path = None
    for p in possible_paths:
        if os.path.exists(p):
            model_path = p
            break
            
    if model_path is None:
        raise FileNotFoundError(f"Checkpoint not found for {algorithm} {dataset} session {session}. Checked: {possible_paths}")

    # 2. Load the checkpoint
    model_data = torch.load(model_path, map_location=device, weights_only=False)

    # 3. Extract the inner state_dict
    if isinstance(model_data, dict):
        if "model_state_dict" in model_data:
            state_dict = model_data["model_state_dict"]
        elif "state_dict" in model_data:
            state_dict = model_data["state_dict"]
        elif "model" in model_data:
            state_dict = model_data["model"]
        else:
            state_dict = model_data
    else:
        state_dict = model_data

    # 4. Instantiate base model architecture
    # (Using your existing model instantiation logic from load_models_fixed.py)
    from models.PyCIL.utils.inc_net import DERNet, FOSTERNet, IncrementalNet, get_convnet
    
    # Ensure valid algorithm_args dictionary exists
    alg_args = getattr(shapArgs, 'algorithm_args', None) if shapArgs is not None else None
    if alg_args is None:
        alg_args = get_algorithm_args(algorithm, dataset)
    if isinstance(alg_args, dict) and "convnet_type" not in alg_args:
        alg_args["convnet_type"] = "resnet18" if dataset in ["cifar100", "imagenet200"] else "resnet18_cbam"
    elif not isinstance(alg_args, dict):
        alg_args = {"convnet_type": "resnet18_cbam", "device": device}
    
    # Calculate classes up to current session
    init_cls = getattr(shapArgs.dataset_params, 'init_cls', 3 if dataset == "dermamnist" else getattr(shapArgs.dataset_params, 'class_per_task', 2)) if shapArgs is not None else 3
    cls_per_task = getattr(shapArgs.dataset_params, 'class_per_task', 2) if shapArgs is not None else 2

    # Detect multi-branch convnets count in state_dict or determine from fc input dimension
    conv_indices = set()
    for k in state_dict.keys():
        if k.startswith("convnets."):
            conv_indices.add(int(k.split(".")[1]))
            
    if conv_indices:
        num_branches = max(len(conv_indices), max(conv_indices) + 1)
    elif "fc.weight" in state_dict and max(state_dict["fc.weight"].shape) > 64:
        feat_dim = max(state_dict["fc.weight"].shape)
        num_branches = max(1, feat_dim // 64) if feat_dim % 64 == 0 else (session + 1)
    else:
        num_branches = session + 1

    # Instantiate model
    if algorithm in ["der", "memo", "ds-al", "dsal", "tagfex"]:
      
        model = DERNet(alg_args, False)
        # Allocate the exact number of active branches present in checkpoint
        model.convnets = nn.ModuleList([_extract_convnet(get_convnet(alg_args, False)) for _ in range(num_branches)])
        
        if "fc.weight" in state_dict:
            out_dim = state_dict["fc.weight"].shape[0]
            in_dim = state_dict["fc.weight"].shape[1]
            model.fc = nn.Linear(in_dim, out_dim, bias=True)
            model.out_dim = in_dim // num_branches

        if "aux_fc.weight" in state_dict:
            aux_out = state_dict["aux_fc.weight"].shape[0]
            aux_in = state_dict["aux_fc.weight"].shape[1]
            model.aux_fc = nn.Linear(aux_in, aux_out, bias=True)
        elif hasattr(model, "aux_fc"):
            model.aux_fc = None

    elif algorithm == "foster":
        model = FOSTERNet(alg_args, False)
        model.convnets = nn.ModuleList([_extract_convnet(get_convnet(alg_args, False)) for _ in range(num_branches)])
        
        if "fc.weight" in state_dict:
            out_dim = state_dict["fc.weight"].shape[0]
            in_dim = state_dict["fc.weight"].shape[1]
            model.fc = nn.Linear(in_dim, out_dim, bias=True)
            model.out_dim = in_dim // num_branches
            
        if "oldfc.weight" in state_dict:
            old_out = state_dict["oldfc.weight"].shape[0]
            old_in = state_dict["oldfc.weight"].shape[1]
            model.oldfc = nn.Linear(old_in, old_out, bias=True)
        elif hasattr(model, "oldfc"):
            model.oldfc = None

        if "fe_fc.weight" in state_dict:
            fe_out = state_dict["fe_fc.weight"].shape[0]
            fe_in = state_dict["fe_fc.weight"].shape[1]
            has_bias = "fe_fc.bias" in state_dict
            model.fe_fc = nn.Linear(fe_in, fe_out, bias=has_bias)
        elif hasattr(model, "fe_fc"):
            model.fe_fc = None
    else:
        # Generic PyCIL loader (iCaRL, SimpleCIL, etc.)
        model = IncrementalNet(alg_args, False)
        if hasattr(model, 'convnet'):
            model.convnet = _extract_convnet(model.convnet)
            
        if "fc.weight" in state_dict:
            out_dim = state_dict["fc.weight"].shape[0]
            in_dim = state_dict["fc.weight"].shape[1]
            model.fc = nn.Linear(in_dim, out_dim, bias=True)
        else:
            total_cls = init_cls + session * cls_per_task
            in_dim = getattr(model.convnet, 'out_dim', 64)
            model.fc = nn.Linear(in_dim, total_cls, bias=True)

    # 5. Load the extracted state_dict into the model
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    return model


#--------------------For iTAML--------------------#
args = iTAMLArgs
use_cuda = True if torch.cuda.is_available() else 'cpu'

def meta_test(model, memory, inc_dataset, testloader):
    all_models = []
    # switch to evaluate mode
    model.eval()

    meta_models = []
    base_model = copy.deepcopy(model)
    class_acc = {}
    meta_task_test_list = {}
    for task_idx in range(args.sess + 1):

        memory_data, memory_target = memory
        memory_data = np.array(memory_data, dtype="int32")
        memory_target = np.array(memory_target, dtype="int32")

        mem_idx = np.where((memory_target >= task_idx * args.class_per_task) & (
                    memory_target < (task_idx + 1) * args.class_per_task))[0]
        meta_memory_data = memory_data[mem_idx]
        meta_memory_target = memory_target[mem_idx]
        meta_model = copy.deepcopy(base_model)

        meta_loader = inc_dataset.get_custom_loader_idx(meta_memory_data, mode="train", batch_size=64)

        meta_optimizer = optim.Adam(meta_model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0,
                                    amsgrad=False)

        meta_model.train()

        # The range of classes that could be predicted in that task
        ai = args.class_per_task * task_idx
        bi = args.class_per_task * (task_idx + 1)
        # Total number of classes learned
        bb = args.class_per_task * (args.sess + 1)
        print("Training meta tasks:\t", task_idx)

        # META training
        if (args.sess != 0):
            for ep in range(1):
                for batch_idx, (inputs, targets) in enumerate(meta_loader):
                    targets_one_hot = torch.FloatTensor(inputs.shape[0], (task_idx + 1) * args.class_per_task)
                    targets_one_hot.zero_()
                    targets_one_hot.scatter_(1, targets[:, None], 1)
                    target_set = np.unique(targets)

                    if use_cuda:
                        inputs, targets_one_hot, targets = inputs.cuda(), targets_one_hot.cuda(), targets.cuda()
                    inputs, targets_one_hot, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(
                        targets_one_hot), torch.autograd.Variable(targets)

                    _, outputs = meta_model(inputs)
                    class_pre_ce = outputs.clone()
                    class_pre_ce = class_pre_ce[:, ai:bi]
                    class_tar_ce = targets_one_hot.clone()

                    loss = F.binary_cross_entropy_with_logits(class_pre_ce, class_tar_ce[:, ai:bi])

                    meta_optimizer.zero_grad()
                    loss.backward()
                    meta_optimizer.step()

        # META testing with given knowledge on task
        meta_model.eval()
        for cl in range(args.class_per_task):
            class_idx = cl + args.class_per_task * task_idx
            loader = inc_dataset.get_custom_loader_class([class_idx], mode="test", batch_size=10)

            for batch_idx, (inputs, targets) in enumerate(loader):
                targets_task = targets - args.class_per_task * task_idx

                if use_cuda:
                    inputs, targets_task = inputs.cuda(), targets_task.cuda()
                inputs, targets_task = torch.autograd.Variable(inputs), torch.autograd.Variable(targets_task)

                _, outputs = meta_model(inputs)

                if use_cuda:
                    inputs, targets = inputs.cuda(), targets_task.cuda()
                inputs, targets_task = torch.autograd.Variable(inputs), torch.autograd.Variable(targets_task)

                pred = torch.argmax(outputs[:, ai:bi], 1, keepdim=False)
                pred = pred.view(1, -1)
                correct = pred.eq(targets_task.view(1, -1).expand_as(pred)).view(-1)

                correct_k = float(torch.sum(correct).detach().cpu().numpy())

                for i, p in enumerate(pred.view(-1)):
                    key = int(p.detach().cpu().numpy())
                    key = key + args.class_per_task * task_idx
                    if (correct[i] == 1):
                        if (key in class_acc.keys()):
                            class_acc[key] += 1
                        else:
                            class_acc[key] = 1

        #           META testing - no knowledge on task
        meta_model.eval()
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

            _, outputs = meta_model(inputs)
            outputs_base, _ = model(inputs)
            task_ids = outputs

            task_ids = task_ids.detach().cpu()
            outputs = outputs.detach().cpu()
            outputs = outputs.detach().cpu()
            outputs_base = outputs_base.detach().cpu()

            bs = inputs.size()[0]
            for i, t in enumerate(list(range(bs))):
                j = batch_idx * args.test_batch + i
                output_base_max = []
                for si in range(args.sess + 1):
                    sj = outputs_base[i][si * args.class_per_task:(si + 1) * args.class_per_task]
                    sq = torch.max(sj)
                    output_base_max.append(sq)

                task_argmax = np.argsort(outputs[i][ai:bi])[-5:]
                task_max = outputs[i][ai:bi][task_argmax]

                if (j not in meta_task_test_list.keys()):
                    meta_task_test_list[j] = [[task_argmax, task_max, output_base_max, targets[i]]]
                else:
                    meta_task_test_list[j].append([task_argmax, task_max, output_base_max, targets[i]])
        if args.sess == args.num_task-1:
            all_models.append(meta_model.to('cpu'))
            # Save the meta_model for each task_idx
            #torch.save(meta_model.state_dict(), f"meta_model_task_{task_idx}_session_{args.sess}.pth")
            #print(f"Saved adapted meta model for Task {task_idx} to {save_path}")
        elif args.sess != args.num_task-1 and task_idx == args.sess:
            all_models.append(meta_model.to('cpu'))
            # Save the adapted model for classes 4 and 5
            #torch.save(meta_model.state_dict(), 'meta_model_task2_classes4_5.pth')
        del meta_model

    '''
    acc_task = {}
    for i in range(args.sess + 1):
        acc_task[i] = 0
        for j in range(args.class_per_task):
            try:
                acc_task[i] += class_acc[i * args.class_per_task + j] / args.sample_per_task_testing[i] * 100
            except:
                pass
    print("\n".join([str(acc_task[k]).format(".4f") for k in acc_task.keys()]))
    print(class_acc)
    '''

    #print("Meta models:")
    #print(all_models)

    #return acc_task
    return all_models

def load_meta_models(dataset, sess):

    args = iTAMLArgs
    if dataset == "cifar100":
        args.class_per_task = 10
        args.num_class = 100
        args.num_task = 10
    else:
        args.class_per_task = 2
        args.num_class = 10
        args.num_task = 5

    args.dataset = dataset
    args.sess = sess
    args.data_path = f"Datasets/{dataset}/"
    inc_dataset = data.IncrementalDataset(
        dataset_name=args.dataset,
        args=args,
        random_order=args.random_classes,
        shuffle=True,
        seed=1,
        batch_size=args.train_batch,
        workers=args.workers,
        validation_split=args.validation,
        increment=args.class_per_task,
    )

    #if (start_sess == ses and start_sess != 0):
    if args.sess != 0:
        inc_dataset._current_task = args.sess
        #with open(f"Saliency/iTAML/{dataset}" + "/sample_per_task_testing_" + str(args.sess - 1) + ".pickle", 'rb') as handle:
        #    sample_per_task_testing = pickle.load(handle)
        #inc_dataset.sample_per_task_testing = sample_per_task_testing
        #args.sample_per_task_testing = sample_per_task_testing

    memory = None
    if args.sess > 0:
        with open(f"saved_models/iTAML/{dataset}" + "/memory_" + str(args.sess - 1) + ".pickle", 'rb') as handle:
            memory = pickle.load(handle)

    _, _, _, testloader, for_memory = inc_dataset.new_task(memory)
    memory = inc_dataset.get_memory(memory, for_memory)
    model = load_model("iTAML", dataset, args.sess, args=args)
    return meta_test(model, memory, inc_dataset, testloader)
