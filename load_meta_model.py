import torch.optim as optim
import pickle
import numpy as np
import copy

from resnet import *
from Saliency.iTAML.radam import *

from Saliency.iTAML import incremental_dataloader as data
from Saliency.imports import iTAMLArgs
from saliency_generator import load_model

sess = 0
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
        if sess == 4:
            all_models.append(meta_model)
            # Save the meta_model for each task_idx
            #torch.save(meta_model.state_dict(), f"meta_model_task_{task_idx}_session_{args.sess}.pth")
            #print(f"Saved adapted meta model for Task {task_idx} to {save_path}")
        elif sess != 4 and task_idx == sess:
            all_models.append(meta_model)
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

    print("Meta models:")
    print(all_models)

    #return acc_task
    return all_models

def load_meta_models(dataset, sess):

    args = iTAMLArgs
    if dataset == "cifar100":
        args.class_per_task = 10
        args.num_class = 100
    else:
        args.class_per_task = 2
        args.num_class = 10

    args.dataset = dataset
    args.sess = sess
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
    if sess != 0:
        inc_dataset._current_task = sess
        #with open(f"Saliency/iTAML/{dataset}" + "/sample_per_task_testing_" + str(args.sess - 1) + ".pickle", 'rb') as handle:
        #    sample_per_task_testing = pickle.load(handle)
        #inc_dataset.sample_per_task_testing = sample_per_task_testing
        #args.sample_per_task_testing = sample_per_task_testing

    memory = None
    if sess > 0:
        with open(f"Saliency/iTAML/{dataset}" + "/memory_" + str(args.sess - 1) + ".pickle", 'rb') as handle:
            memory = pickle.load(handle)

    _, _, _, testloader, for_memory = inc_dataset.new_task(memory)
    memory = inc_dataset.get_memory(memory, for_memory)
    model = load_model("iTAML", dataset, sess, args=args)
    return meta_test(model, memory, inc_dataset, testloader)

if __name__ == '__main__':
    dataset = "cifar10"
    sess = 0
    load_meta_models(dataset, sess)
