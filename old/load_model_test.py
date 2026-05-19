import os
import torchvision.datasets as datasets
import argparse

from Saliency.PyCIL.utils.toolkit import tensor2numpy

# Saliency Imports
from torch.utils.data import Subset, DataLoader
import saliency_dataloader as sdl
from saliency_generator import load_model as sg_load_model

# Imports needed for each model
from Saliency.imports import *

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def compute_accuracy(predictions, targets):
    correct, total = 0, 0
    correct += (predictions.cpu() == targets).sum()
    total += len(targets)
    correct = correct.cpu().data.numpy()
    return np.around(correct * 100 / total, decimals=2)

def generate_predictions(model, ses, images, labels=None, **kwargs):
    if SalGenArgs.algorithm == "xder":
        outputs = model(images)
        _, pred = torch.max(outputs[:, :SalGenArgs.class_per_task * (1 + ses)].data, 1)
    else:
        with torch.no_grad():
            outputs = model(images)["logits"]
        pred = torch.max(outputs, dim=1)[1]

    predicted = pred.squeeze()
    if labels is not None:
        acc = compute_accuracy(predicted, labels)
        print("Acc:", acc)

    return predicted


def get_indices(dataset, class_name):
    indices = []
    for i in range(len(dataset.targets)):
        for j in class_name:
            if dataset.targets[i] == j:
                indices.append(i)
    return indices


def load_model(algorithm, dataset, ses, **kwargs):
    model = None
    model_path = ""

    model_path = f"Saliency/{algorithm}/{dataset}/{algorithm}_ses_{ses}.pth"
    #model = factory.get_model(algorithm, fosterArgs)._network
    match algorithm:
        case "foster":
            model = FOSTERNet(fosterArgs, False)
        
            #Session 0 Accuracy: 15.0
            #Session 1 Accuracy: 5.9
            #Session 2 Accuracy: 3.23
            #Session 3 Accuracy: 2.33
            #Session 4 Accuracy: 0.72
            #Session 5 Accuracy: 0.6
        case "memo":
            model = AdaptiveNet(memoArgs, False)
            
            #Session 0 Accuracy: 15.0
            #Session 1 Accuracy: 5.6
            #Session 2 Accuracy: 3.03
            #Session 3 Accuracy: 2.2
            #Session 4 Accuracy: 0.96
            #Session 5 Accuracy: 0.48
        case "der":
            model = DERNet(derArgs, False)
            
            #Session 0 Accuracy: 15.5
            #Session 1 Accuracy: 5.7
            #Session 2 Accuracy: 3.47
            #Session 3 Accuracy: 2.48
            #Session 4 Accuracy: 0.88
            #Session 5 Accuracy: 0.58
        case "tagfex":
            model = TagFexNet(tagfexArgs, False)
    # Update the model architecture to match the task
    if ses > 0:
        start = 0 if algorithm != "foster" else ses-1
        for i in range(start, ses + 1):
            model.update_fc(2 * (i + 1))
    else: model.update_fc(2 * (ses + 1))


    model_data = torch.load(model_path, map_location=device, weights_only=False)

    if algorithm == "RPSnet":
        model.load_state_dict(model_data['state_dict'])
    else:
        model.load_state_dict(model_data)

    model.eval()

    return model

def load_xder(dataset, ses):
    model = None
    model_path = ""

    #model_path = f"Saliency/xder/{dataset}/xder_seq-{dataset}_{ses}.pt"
    #model_path = f"./{dataset}/xder_seq-{dataset}_{ses}.pt"
    model_path = f"./cifar10/xder_ses_{ses}.pt"

    #model_data = torch.load(model_path, map_location=device, weights_only=False)

    xderArgs['dataset'] = f'seq-{dataset}'
    args = argparse.Namespace(**xderArgs)

    xder_dataset = get_dataset_class(args)
    backbone_cl, backbone_args = get_backbone_class('resnet18', return_args=True)
    parsed_args = {arg: getattr(args, arg) for arg in backbone_args.keys()}
    model = XDer(backbone_cl(**parsed_args),
                 xder_dataset.get_loss(),
                 args,
                 xder_dataset.get_transform(),
                 dataset=xder_dataset)

    model, _ = mammoth_load_checkpoint(model_path, model)

    model.eval()

    return model

def load_saliency_data(dataset, desired_classes, imgs_per_class):
    transform = transforms.Compose(
        [transforms.ToTensor()])

    if not os.path.isdir(f"SaliencyMaps/{SalGenArgs.algorithm}/" + SalGenArgs.dataset):
        os.makedirs(f"SaliencyMaps/{SalGenArgs.algorithm}/" + SalGenArgs.dataset)

    saliency_set = torch.utils.data.Dataset()
    mean = None
    std = None

    test_trsf = [transforms.ToTensor()]
    common_trsf = [transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)), ]
    trsf = transforms.Compose([*test_trsf, *common_trsf])

    saliency_set = datasets.cifar.CIFAR100(f"Datasets/{dataset}/", train=False, download=True, transform=trsf)
    #test_data, test_targets = saliency_set.data, np.array(saliency_set.targets)
    mean = torch.tensor([0.5071, 0.4867, 0.4408])
    std = torch.tensor([0.2675, 0.2565, 0.2761])

    idx = get_indices(saliency_set, desired_classes)

    subset = Subset(saliency_set, idx)

    # Create a DataLoader for the subset
    saliencyLoader = DataLoader(subset, batch_size=4096)

    images, labels = next(iter(saliencyLoader))

    # Order saliency images and labels by class
    sal_idx = []
    sal_labels = []
    for i in range(len(desired_classes)):
        num = 0
        while len(sal_idx) < imgs_per_class * (i + 1):
            if labels[num] == desired_classes[i]:
                sal_idx.append(num)
                sal_labels.append(desired_classes[i])
            #num += 4
            #num += 16
            num += 1
    sal_imgs = images[sal_idx]

    return sal_imgs, torch.tensor(sal_labels), saliency_set.classes, mean, std


### From PyCIL ###
import logging
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from Saliency.PyCIL.utils.data import iCIFAR10, iCIFAR100, iImageNet100, iImageNet1000
from tqdm import tqdm

class DataManager(object):
    def __init__(self, dataset_name, shuffle, seed, init_cls, increment):
        self.dataset_name = dataset_name
        self._setup_data(dataset_name, shuffle, seed)
        assert init_cls <= len(self._class_order), "No enough classes."
        self._increments = [init_cls]
        while sum(self._increments) + increment < len(self._class_order):
            self._increments.append(increment)
        offset = len(self._class_order) - sum(self._increments)
        if offset > 0:
            self._increments.append(offset)

    @property
    def nb_tasks(self):
        return len(self._increments)

    def get_task_size(self, task):
        return self._increments[task]

    def get_accumulate_tasksize(self, task):
        return sum(self._increments[:task + 1])

    def get_total_classnum(self):
        return len(self._class_order)

    def get_dataset(
            self, indices, source, mode, appendent=None, ret_data=False, m_rate=None
    ):
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "flip":
            trsf = transforms.Compose(
                [
                    *self._test_trsf,
                    transforms.RandomHorizontalFlip(p=1.0),
                    *self._common_trsf,
                ]
            )
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        data, targets = [], []
        for idx in indices:
            if m_rate is None:
                class_data, class_targets = self._select(
                    x, y, low_range=idx, high_range=idx + 1
                )
            else:
                class_data, class_targets = self._select_rmm(
                    x, y, low_range=idx, high_range=idx + 1, m_rate=m_rate
                )
            data.append(class_data)
            targets.append(class_targets)

        if appendent is not None and len(appendent) != 0:
            appendent_data, appendent_targets = appendent
            data.append(appendent_data)
            targets.append(appendent_targets)

        data, targets = np.concatenate(data), np.concatenate(targets)

        if ret_data:
            return data, targets, DummyDataset(data, targets, trsf, self.use_path)
        else:
            return DummyDataset(data, targets, trsf, self.use_path)

    def get_finetune_dataset(self, known_classes, total_classes, source, mode, appendent, type="ratio"):
        if source == 'train':
            x, y = self._train_data, self._train_targets
        elif source == 'test':
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError('Unknown data source {}.'.format(source))

        if mode == 'train':
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == 'test':
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError('Unknown mode {}.'.format(mode))
        val_data = []
        val_targets = []

        old_num_tot = 0
        appendent_data, appendent_targets = appendent

        for idx in range(0, known_classes):
            append_data, append_targets = self._select(appendent_data, appendent_targets,
                                                       low_range=idx, high_range=idx + 1)
            num = len(append_data)
            if num == 0:
                continue
            old_num_tot += num
            val_data.append(append_data)
            val_targets.append(append_targets)
        if type == "ratio":
            new_num_tot = int(old_num_tot * (total_classes - known_classes) / known_classes)
        elif type == "same":
            new_num_tot = old_num_tot
        else:
            assert 0, "not implemented yet"
        new_num_average = int(new_num_tot / (total_classes - known_classes))
        for idx in range(known_classes, total_classes):
            class_data, class_targets = self._select(x, y, low_range=idx, high_range=idx + 1)
            val_indx = np.random.choice(len(class_data), new_num_average, replace=False)
            val_data.append(class_data[val_indx])
            val_targets.append(class_targets[val_indx])
        val_data = np.concatenate(val_data)
        val_targets = np.concatenate(val_targets)
        return DummyDataset(val_data, val_targets, trsf, self.use_path)

    def get_dataset_with_split(
            self, indices, source, mode, appendent=None, val_samples_per_class=0
    ):
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        train_data, train_targets = [], []
        val_data, val_targets = [], []
        for idx in indices:
            class_data, class_targets = self._select(
                x, y, low_range=idx, high_range=idx + 1
            )
            val_indx = np.random.choice(
                len(class_data), val_samples_per_class, replace=False
            )
            train_indx = list(set(np.arange(len(class_data))) - set(val_indx))
            val_data.append(class_data[val_indx])
            val_targets.append(class_targets[val_indx])
            train_data.append(class_data[train_indx])
            train_targets.append(class_targets[train_indx])

        if appendent is not None:
            appendent_data, appendent_targets = appendent
            for idx in range(0, int(np.max(appendent_targets)) + 1):
                append_data, append_targets = self._select(
                    appendent_data, appendent_targets, low_range=idx, high_range=idx + 1
                )
                val_indx = np.random.choice(
                    len(append_data), val_samples_per_class, replace=False
                )
                train_indx = list(set(np.arange(len(append_data))) - set(val_indx))
                val_data.append(append_data[val_indx])
                val_targets.append(append_targets[val_indx])
                train_data.append(append_data[train_indx])
                train_targets.append(append_targets[train_indx])

        train_data, train_targets = np.concatenate(train_data), np.concatenate(
            train_targets
        )
        val_data, val_targets = np.concatenate(val_data), np.concatenate(val_targets)

        return DummyDataset(
            train_data, train_targets, trsf, self.use_path
        ), DummyDataset(val_data, val_targets, trsf, self.use_path)

    def _setup_data(self, dataset_name, shuffle, seed):
        idata = _get_idata(dataset_name)
        idata.download_data()

        # Data
        self._train_data, self._train_targets = idata.train_data, idata.train_targets
        self._test_data, self._test_targets = idata.test_data, idata.test_targets
        self.use_path = idata.use_path

        # Transforms
        self._train_trsf = idata.train_trsf
        self._test_trsf = idata.test_trsf
        self._common_trsf = idata.common_trsf

        # Order
        order = [i for i in range(len(np.unique(self._train_targets)))]
        if shuffle:
            np.random.seed(seed)
            order = np.random.permutation(len(order)).tolist()
        else:
            order = idata.class_order
        self._class_order = order
        logging.info(self._class_order)

        # Map indices
        self._train_targets = _map_new_class_index(
            self._train_targets, self._class_order
        )
        self._test_targets = _map_new_class_index(self._test_targets, self._class_order)

    def _select(self, x, y, low_range, high_range):
        idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]

        if isinstance(x, np.ndarray):
            x_return = x[idxes]
        else:
            x_return = []
            for id in idxes:
                x_return.append(x[id])
        return x_return, y[idxes]

    def _select_rmm(self, x, y, low_range, high_range, m_rate):
        assert m_rate is not None
        if m_rate != 0:
            idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
            selected_idxes = np.random.randint(
                0, len(idxes), size=int((1 - m_rate) * len(idxes))
            )
            new_idxes = idxes[selected_idxes]
            new_idxes = np.sort(new_idxes)
        else:
            new_idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[new_idxes], y[new_idxes]

    def getlen(self, index):
        y = self._train_targets
        return np.sum(np.where(y == index))


class DummyDataset(Dataset):
    def __init__(self, images, labels, trsf, use_path=False):
        assert len(images) == len(labels), "Data size error!"
        self.images = images
        self.labels = labels
        self.trsf = trsf
        self.use_path = use_path

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.use_path:
            image = self.trsf(pil_loader(self.images[idx]))
        else:
            image = self.trsf(Image.fromarray(self.images[idx]))
        label = self.labels[idx]

        return idx, image, label


def _map_new_class_index(y, order):
    return np.array(list(map(lambda x: order.index(x), y)))


def _get_idata(dataset_name):
    name = dataset_name.lower()
    if name == "cifar10":
        return iCIFAR10()
    elif name == "cifar100":
        return iCIFAR100()
    elif name == "imagenet1000":
        return iImageNet1000()
    elif name == "imagenet100":
        return iImageNet100()
    else:
        raise NotImplementedError("Unknown dataset {}.".format(dataset_name))


def pil_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def accimage_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    accimage is an accelerated Image loader and preprocessor leveraging Intel IPP.
    accimage is available on conda-forge.
    """
    import accimage

    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)


def accuracy(y_pred, y_true, nb_old, increment=10):
    assert len(y_pred) == len(y_true), "Data length error."
    all_acc = {}
    all_acc["total"] = np.around(
        (y_pred == y_true).sum() * 100 / len(y_true), decimals=2
    )

    # Grouped accuracy
    for class_id in range(0, np.max(y_true), increment):
        idxes = np.where(
            np.logical_and(y_true >= class_id, y_true < class_id + increment)
        )[0]
        label = "{}-{}".format(
            str(class_id).rjust(2, "0"), str(class_id + increment - 1).rjust(2, "0")
        )
        all_acc[label] = np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
        )

    # Old accuracy
    idxes = np.where(y_true < nb_old)[0]
    all_acc["old"] = (
        0
        if len(idxes) == 0
        else np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
        )
    )

    # New accuracy
    idxes = np.where(y_true >= nb_old)[0]
    all_acc["new"] = np.around(
        (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
    )

    return all_acc


SalGenArgs.algorithm = "der"
SalGenArgs.dataset = "imagenet200"
if SalGenArgs.dataset == "cifar100":
    SalGenArgs.class_per_task = 10
    SalGenArgs.num_class = 100
elif SalGenArgs.dataset == "imagenet200":
    SalGenArgs.class_per_task = 20
    SalGenArgs.num_class = 200
else:
    SalGenArgs.class_per_task = 2
    SalGenArgs.num_class = 10
salDataloader = sdl.SalDataloader(SalGenArgs)

for task in range(10):
#for task in range(5):
    dataset = "imagenet200"
    #model = load_xder(dataset, task)
    model = sg_load_model("der", dataset, task)
    print(f"Finished Loading Model {task}.")

    params = ((0, [0, 1]),
              (1, [2, 3]),
              (2, [4, 5]),
              (3, [6, 7]),
              (4, [8, 9]))


    sal_imgs, sal_labels, classes, MEAN, STD = salDataloader.load_data(range(task * 20, (task * 20) + 20),
                                                                       5, batch_size=10000)
    sal_imgs, sal_labels = sal_imgs.to(device), sal_labels.to(device)
    generate_predictions(model, task, sal_imgs, sal_labels)

    '''
    data_manager = DataManager(
        dataset,
        fosterArgs["shuffle"],
        fosterArgs["seed"],
        fosterArgs["init_cls"],
        fosterArgs["increment"],
    )

    test_dataset = data_manager.get_dataset(
        np.arange(0, 10+10*task), source="test", mode="test"
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=fosterArgs["batch_size"],
        shuffle=False,
        #num_workers=fosterArgs["num_workers"],
    )
    '''

    '''y_pred, y_true = [], []
    for _, (_, inputs, targets) in enumerate(test_loader):
        inputs = inputs.to("cpu")
        with torch.no_grad():
            outputs = model(inputs)["logits"]
        predicts = torch.topk(
            outputs, k=5, dim=1, largest=True, sorted=True
        )[
            1
        ]  # [bs, topk]
        y_pred.append(predicts.cpu().numpy())
        y_true.append(targets.cpu().numpy())

    np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]

    ret = {}
    grouped = accuracy(y_pred.T[0], y_true, 10+10*task)
    ret["grouped"] = grouped
    ret["top1"] = grouped["total"]
    ret["top{}".format(5)] = np.around(
        (y_pred.T == np.tile(y_true, (5, 1))).sum() * 100 / len(y_true),
        decimals=2,
    )'''

    '''
    #cnn_accy = ret
    #print(cnn_accy)
    correct, total = 0, 0
    for i, (_, inputs, targets) in enumerate(test_loader):
        inputs = inputs.to("cpu")
        with torch.no_grad():
            outputs = model(inputs)["logits"]
        predicts = torch.max(outputs, dim=1)[1]
        correct += (predicts.cpu() == targets).sum()
        total += len(targets)

    print(f"Session {task} Accuracy: {np.around(tensor2numpy(correct) * 100 / total, decimals=2)}")
    #print("\n\n")
    '''