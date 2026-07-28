import argparse
import copy
import os
import pickle
import numpy as np
import torch
import torch.optim as optim

from models.iTAML import incremental_dataloader as data
from models.iTAML.radam import *
from models.iTAML.resnet import *
from utils.model_parameters import (
    AdaptiveNet,
    BasicNet1,
    DERNet,
    DSALNet,
    FOSTERNet,
    IncrementalNet,
    RPS_net_cifar,
    RPS_net_mlp,
    TagFexNet,
    XDer,
    get_algorithm_args,
    get_backbone_class,
    get_dataset_class,
    iTAMLArgs,
    mammoth_load_checkpoint,
    original_cwd,
    pycil_algs,
)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
DRIVE_BASE_DIR = "/content/drive/MyDrive/CL-SHAPC-Interpretability/savedmodels"


def compute_accuracy(predictions, targets):
  correct, total = 0, 0
  correct += (predictions.cpu() == targets).sum()
  total += len(targets)
  correct = correct.cpu().data.numpy()
  return np.around(correct * 100 / total, decimals=2)


def generate_predictions(algorithm, model, ses, images, labels=None, **kwargs):
  images = images.to(device)

  if algorithm == "iTAML":
    model.set_shap(True)
    outputs2 = model(images)
    pred = torch.argmax(
        outputs2[:, 0 : kwargs["cls_per_task"] * (1 + ses)], 1, keepdim=False
    )
  elif algorithm == "RPSnet":
    outputs = model(images, kwargs["infer_path"], -1)
    _, pred = torch.max(outputs, 1)
  elif algorithm == "DGR":
    real_scores = model.forward(images)
    _, pred = real_scores[:, 0 : kwargs["cls_per_task"] * (1 + ses)].max(1)
  elif algorithm in pycil_algs:
    with torch.no_grad():
      outputs = model(images)
      logits = outputs["logits"] if isinstance(outputs, dict) else outputs
    pred = torch.max(logits, dim=1)[1]
  elif algorithm == "xder":
    outputs = model(images)
    _, pred = torch.max(
        outputs[:, : kwargs["cls_per_task"] * (1 + ses)].data, 1
    )
  predicted = pred.squeeze()
  if labels is not None:
    acc = compute_accuracy(predicted, labels)
    print("Acc:", acc)

  return predicted


def load_model(algorithm, dataset, ses, shapArgs=None):
  model = None
  scholar = None
  model_path = ""
  alg_args = get_algorithm_args(algorithm, dataset)

  if alg_args is None:
    alg_args = {
        "convnet_type": (
            "resnet32" if "mnist" in dataset.lower() else "resnet18"
        ),
        "device": [device],
        "dataset": dataset,
    }

  if algorithm in pycil_algs:
    model_path = os.path.join(
        DRIVE_BASE_DIR, algorithm, dataset, f"{algorithm}_ses_{ses}.pth"
    )
    if not os.path.exists(model_path):
      model_path = (
          f"savedmodels/{algorithm}/{dataset}/{algorithm}_ses_{ses}.pth"
      )

    if not os.path.exists(model_path):
      raise FileNotFoundError(
          f"Could not locate checkpoint file at: {model_path}"
      )

    model_data = torch.load(model_path, map_location=device, weights_only=False)

    # Key remapping for DERNet
    if algorithm == "der" and isinstance(model_data, dict):
      renamed_data = {}
      for k, v in model_data.items():
        if k.startswith("fe_fc."):
          renamed_data[k.replace("fe_fc.", "aux_fc.")] = v
        else:
          renamed_data[k] = v
      model_data = renamed_data

    match algorithm:
      case "foster":
        model = FOSTERNet(alg_args, False)
      case "memo":
        model = AdaptiveNet(alg_args, False)
      case "der":
        model = DERNet(alg_args, False)
      case "icarl":
        model = IncrementalNet(alg_args, False)
      case "ds-al":
        configs = alg_args.get("configurations", {}) if alg_args else {}
        ds_cfg = configs.get(dataset, {}) if configs else {}

        buffer_size = ds_cfg.get("buffer_size", 8192)
        gamma = ds_cfg.get("gamma", 1e-3)
        gamma_comp = ds_cfg.get("gamma_comp", 1e-3)
        compensation_ratio = ds_cfg.get("compensation_ratio", 1.0)

        model = DSALNet(
            alg_args,
            buffer_size=buffer_size,
            gamma_main=gamma,
            gamma_comp=gamma_comp,
            C=compensation_ratio,
        )
        model.generate_buffer()
        model.generate_fc()
      case "tagfex":
        model = TagFexNet(alg_args, False)

    ###--- Progressive Architecture Alignment & Attribute Fallbacks ---###
    init_cls = (
        getattr(shapArgs.dataset_params, "init_cls", None) if shapArgs else None
    )
    cls_per_task = (
        getattr(shapArgs.dataset_params, "class_per_task", None)
        if shapArgs
        else None
    )

    if init_cls is None:
      init_cls = (
          cls_per_task
          if cls_per_task is not None
          else alg_args.get("init_cls", 3)
      )

    if cls_per_task is None:
      cls_per_task = alg_args.get("increment", 2)

    if algorithm == "foster":
      current_total_cls = init_cls
      model.update_fc(current_total_cls)
      for s in range(1, ses + 1):
        model.copy()
        current_total_cls += cls_per_task
        model.update_fc(current_total_cls)
    else:
      current_total_cls = init_cls
      model.update_fc(current_total_cls)
      for s in range(1, ses + 1):
        current_total_cls += cls_per_task
        model.update_fc(current_total_cls)

    # Trimming backbones and overriding linear head dimensions directly from checkpoint
    if isinstance(model_data, dict):
      convnet_indices = [
          int(k.split(".")[1])
          for k in model_data.keys()
          if k.startswith("convnets.") and k.split(".")[1].isdigit()
      ]
      if convnet_indices:
        target_convnets = max(convnet_indices) + 1
        if hasattr(model, "convnets"):
          while (
              len(model.convnets) > target_convnets and len(model.convnets) > 1
          ):
            del model.convnets[-1]

      # DSAL / ACIL Buffer Shape Alignment
      if "buffer.weight" in model_data:
        buf_out_dim, buf_in_dim = model_data["buffer.weight"].shape
        if hasattr(model, "convnet") and hasattr(model.convnet, "out_dim"):
          model.convnet.out_dim = buf_in_dim
        model.buffer_size = buf_out_dim
        if hasattr(model, "generate_buffer"):
          model.generate_buffer()

      # RecursiveLinear Head Alignment (for DSAL / ACIL)
      if algorithm in ["ds-al", "acil"]:
        if "fc.weight" in model_data:
          num_classes = model_data["fc.weight"].shape[1]
          buf_size = model_data["fc.weight"].shape[0]
          if hasattr(model, "fc") and model.fc is not None:
            model.fc.weight = torch.nn.Parameter(
                torch.zeros(
                    buf_size,
                    num_classes,
                    dtype=model.fc.weight.dtype,
                    device=device,
                )
            )
          if hasattr(model, "fc_comp") and model.fc_comp is not None:
            model.fc_comp.weight = torch.nn.Parameter(
                torch.zeros(
                    buf_size,
                    num_classes,
                    dtype=model.fc_comp.weight.dtype,
                    device=device,
                )
            )

      # Standard nn.Linear Head Alignment (for other PyCIL algorithms)
      else:
        if "fc.weight" in model_data:
          fc_out_dim, fc_in_dim = model_data["fc.weight"].shape
          model.fc = torch.nn.Linear(fc_in_dim, fc_out_dim)

        if "aux_fc.weight" in model_data:
          aux_out_dim, aux_in_dim = model_data["aux_fc.weight"].shape
          model.aux_fc = torch.nn.Linear(aux_in_dim, aux_out_dim)

        if "fe_fc.weight" in model_data:
          fe_out_dim, fe_in_dim = model_data["fe_fc.weight"].shape
          model.fe_fc = torch.nn.Linear(fe_in_dim, fe_out_dim)

        if "oldfc.weight" in model_data:
          old_out_dim, old_in_dim = model_data["oldfc.weight"].shape
          model.oldfc = torch.nn.Linear(old_in_dim, old_out_dim)

        if "trans_classifier.weight" in model_data:
          tc_out_dim, tc_in_dim = model_data["trans_classifier.weight"].shape
          if (
              hasattr(model, "generate_fc")
              and getattr(model, "trans_classifier", None) is not None
          ):
            model.trans_classifier = model.generate_fc(tc_in_dim, tc_out_dim)
          else:
            model.trans_classifier = torch.nn.Linear(tc_in_dim, tc_out_dim)

    if hasattr(model, "feature_dim"):
      model.out_dim = int(model.feature_dim)
    ###----------------------------------------------------------------###
  else:
    match algorithm:
      case "iTAML":
        model_path = os.path.join(
            DRIVE_BASE_DIR, algorithm, dataset, f"session_{ses}_model_best.pth.tar"
        )
        if not os.path.exists(model_path):
          model_path = (
              f"Saliency/{algorithm}/{dataset}/session_{ses}_model_best.pth.tar"
          )
        model = BasicNet1(alg_args, 0, device=device)
      case "RPSnet":
        model_path = os.path.join(
            DRIVE_BASE_DIR,
            algorithm,
            dataset,
            f"session_{ses}_0_model_best.pth.tar",
        )
        if not os.path.exists(model_path):
          model_path = (
              f"Saliency/{algorithm}/{dataset}/session_{ses}_0_model_best.pth.tar"
          )
        if dataset == "mnist":
          model = RPS_net_mlp(alg_args)
        else:
          model = RPS_net_cifar(alg_args)
      case "xder":
        model_path = os.path.join(
            DRIVE_BASE_DIR, algorithm, dataset, f"xder_seq-{dataset}_ses_{ses}.pt"
        )
        if not os.path.exists(model_path):
          model_path = f"./{dataset}/xder_seq-{dataset}_ses_{ses}.pt"

        alg_args["dataset"] = f"seq-{dataset}"
        if shapArgs is not None:
          alg_args["num_classes"] = shapArgs.dataset_params.num_class
        args = argparse.Namespace(**alg_args)

        xder_dataset = get_dataset_class(args)
        backbone_cl, backbone_args = get_backbone_class(
            "resnet18", return_args=True
        )
        parsed_args = {
            arg: getattr(args, arg) for arg in backbone_args.keys()
        }
        model = XDer(
            backbone_cl(**parsed_args),
            xder_dataset.get_loss(),
            args,
            xder_dataset.get_transform(),
            dataset=xder_dataset,
        )

        model, _ = mammoth_load_checkpoint(model_path, model)
        model.eval()
        os.chdir(original_cwd)
        return model

    if not os.path.exists(model_path):
      raise FileNotFoundError(
          f"Could not locate checkpoint file at: {model_path}"
      )

    model_data = torch.load(model_path, map_location=device, weights_only=False)

  if algorithm == "RPSnet":
    model.load_state_dict(model_data["state_dict"])
    model.eval()
  elif algorithm == "DGR" and scholar is not None:
    scholar.load_state_dict(model_data["state"])
    model = scholar.solver
  else:
    model.load_state_dict(model_data, strict=False)
    model.eval()

  return model


# --------------------For iTAML--------------------#
args = iTAMLArgs
use_cuda = True if torch.cuda.is_available() else "cpu"


def meta_test(model, memory, inc_dataset, testloader):
  all_models = []
  model.eval()

  meta_models = []
  base_model = copy.deepcopy(model)
  class_acc = {}
  meta_task_test_list = {}
  for task_idx in range(args.sess + 1):

    memory_data, memory_target = memory
    memory_data = np.array(memory_data, dtype="int32")
    memory_target = np.array(memory_target, dtype="int32")

    mem_idx = np.where(
        (memory_target >= task_idx * args.class_per_task)
        & (memory_target < (task_idx + 1) * args.class_per_task)
    )[0]
    meta_memory_data = memory_data[mem_idx]
    meta_memory_target = memory_target[mem_idx]
    meta_model = copy.deepcopy(base_model)

    meta_loader = inc_dataset.get_custom_loader_idx(
        meta_memory_data, mode="train", batch_size=64
    )

    meta_optimizer = optim.Adam(
        meta_model.parameters(),
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0.0,
        amsgrad=False,
    )

    meta_model.train()

    ai = args.class_per_task * task_idx
    bi = args.class_per_task * (task_idx + 1)
    bb = args.class_per_task * (args.sess + 1)
    print("Training meta tasks:\t", task_idx)

    if args.sess != 0:
      for ep in range(1):
        for batch_idx, (inputs, targets) in enumerate(meta_loader):
          targets_one_hot = torch.FloatTensor(
              inputs.shape[0], (task_idx + 1) * args.class_per_task
          )
          targets_one_hot.zero_()
          targets_one_hot.scatter_(1, targets[:, None], 1)

          if use_cuda:
            inputs, targets_one_hot, targets = (
                inputs.cuda(),
                targets_one_hot.cuda(),
                targets.cuda(),
            )
          inputs, targets_one_hot, targets = (
              torch.autograd.Variable(inputs),
              torch.autograd.Variable(targets_one_hot),
              torch.autograd.Variable(targets),
          )

          _, outputs = meta_model(inputs)
          class_pre_ce = outputs.clone()
          class_pre_ce = class_pre_ce[:, ai:bi]
          class_tar_ce = targets_one_hot.clone()

          loss = F.binary_cross_entropy_with_logits(
              class_pre_ce, class_tar_ce[:, ai:bi]
          )

          meta_optimizer.zero_grad()
          loss.backward()
          meta_optimizer.step()

    meta_model.eval()
    for cl in range(args.class_per_task):
      class_idx = cl + args.class_per_task * task_idx
      loader = inc_dataset.get_custom_loader_class(
          [class_idx], mode="test", batch_size=10
      )

      for batch_idx, (inputs, targets) in enumerate(loader):
        targets_task = targets - args.class_per_task * task_idx

        if use_cuda:
          inputs, targets_task = inputs.cuda(), targets_task.cuda()
        inputs, targets_task = (
            torch.autograd.Variable(inputs),
            torch.autograd.Variable(targets_task),
        )

        _, outputs = meta_model(inputs)

        if use_cuda:
          inputs, targets = inputs.cuda(), targets_task.cuda()
        inputs, targets_task = (
            torch.autograd.Variable(inputs),
            torch.autograd.Variable(targets_task),
        )

        pred = torch.argmax(outputs[:, ai:bi], 1, keepdim=False)
        pred = pred.view(1, -1)
        correct = pred.eq(targets_task.view(1, -1).expand_as(pred)).view(-1)

        for i, p in enumerate(pred.view(-1)):
          key = int(p.detach().cpu().numpy())
          key = key + args.class_per_task * task_idx
          if correct[i] == 1:
            class_acc[key] = class_acc.get(key, 0) + 1

    meta_model.eval()
    for batch_idx, (inputs, targets) in enumerate(testloader):
      if use_cuda:
        inputs, targets = inputs.cuda(), targets.cuda()
      inputs, targets = (
          torch.autograd.Variable(inputs),
          torch.autograd.Variable(targets),
      )

      _, outputs = meta_model(inputs)
      outputs_base, _ = model(inputs)

      outputs = outputs.detach().cpu()
      outputs_base = outputs_base.detach().cpu()

      bs = inputs.size()[0]
      for i, t in enumerate(list(range(bs))):
        j = batch_idx * args.test_batch + i
        output_base_max = []
        for si in range(args.sess + 1):
          sj = outputs_base[i][
              si * args.class_per_task : (si + 1) * args.class_per_task
          ]
          sq = torch.max(sj)
          output_base_max.append(sq)

        task_argmax = np.argsort(outputs[i][ai:bi])[-5:]
        task_max = outputs[i][ai:bi][task_argmax]

        if j not in meta_task_test_list.keys():
          meta_task_test_list[j] = [
              [task_argmax, task_max, output_base_max, targets[i]]
          ]
        else:
          meta_task_test_list[j].append(
              [task_argmax, task_max, output_base_max, targets[i]]
          )

    if args.sess == args.num_task - 1 or task_idx == args.sess:
      all_models.append(meta_model.to("cpu"))
    del meta_model

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

  if args.sess != 0:
    inc_dataset._current_task = args.sess

  memory = None
  if args.sess > 0:
    memory_path = os.path.join(
        DRIVE_BASE_DIR, "iTAML", dataset, f"memory_{args.sess - 1}.pickle"
    )
    if not os.path.exists(memory_path):
      memory_path = f"saved_models/iTAML/{dataset}/memory_{args.sess - 1}.pickle"
    with open(memory_path, "rb") as handle:
      memory = pickle.load(handle)

  _, _, _, testloader, for_memory = inc_dataset.new_task(memory)
  memory = inc_dataset.get_memory(memory, for_memory)
  model = load_model("iTAML", dataset, args.sess)
  return meta_test(model, memory, inc_dataset, testloader)
