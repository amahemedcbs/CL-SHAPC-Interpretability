% Setup file
algs = ["iTAML", "RPSnet", "DGR", "foster", "memo", "der", "icarl", "dsal", "tagfex", "xder"];

% Dataset configs
dataset_configs = {};
dataset_configs.cifar10 = struct('num_sessions', 10, 'num_classes', 10, 'cls_per_task', 2, 'samples_per_cls', 100);
dataset_configs.cifar100 = struct('num_sessions', 10, 'num_classes', 100', 'cls_per_task', 10, 'samples_per_cls', 20);
dataset_configs.imagenet200 = struct('num_sessions', 10, 'num_classes', 200', 'cls_per_task', 20, 'samples_per_cls', 20);
