function [threshold] = shapcthresholdtable(algs, dataset)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
arguments (Input)
    algs
    dataset
end

arguments (Output)
    threshold
end

if dataset == "cifar100"
    num_sessions = 10;
else
    num_sessions = 5;
end

shapc_path = sprintf("%s_shapc_data.mat", dataset);

if isfile(shapc_path)
    sh_nm = fieldnames(load(shapc_path));
    shapc_data = load(shapc_path).(string(sh_nm));
else
    shapc_data = struct();
end

all_shapc_vars = [];
all_shapc_stds = [];
means = [];
thresholds = zeros(length(algs), 1);

% Load SHAPC values
for j=1:length(algs)
    alg = algs(j);
    save_path = sprintf("%s/%s/shapc_vals_first_last_1000.mat", alg, dataset);
    if isfile(save_path)
        shapc_struct = load(save_path);
       
        % Load SHAPC values all
        alg_shapc_vars = [];
        shapc_vars = [];
        for i=1:num_sessions-1
            pair_str = 'sc' + string(i-1) +string(num_sessions-1);
            shapcs = [];
            sample_list = string(fieldnames(shapc_struct.(pair_str)));
            for k=1:length(fieldnames(shapc_struct.(pair_str)))
                sample_str = sample_list(k);
                shapcs = [shapcs; shapc_struct.(pair_str).(sample_str)];
            end
            rel_shapc_std = std(shapcs) / mean(shapcs);
            alg_shapc_vars = [alg_shapc_vars; rel_shapc_std];

        end
        
        % Note the shapc is represented as percentage
        %disp("From averaging all tasks together:")
        shapc_var = mean(alg_shapc_vars);
        shapc_var_perc = mean(alg_shapc_vars) * 100;
    else
        shapc_var = NaN;
        shapc_var_perc = NaN;
    end
    all_shapc_vars = [all_shapc_vars; shapc_var_perc];
    all_shapc_stds = [all_shapc_stds; sqrt(shapc_var_perc)];

    shapc_mean = shapc_data.(alg).first_last_1000_shapc;
    means = [means; shapc_mean];
    alg_threshold = shapc_mean - sqrt(shapc_var_perc);
    thresholds(j) = alg_threshold;
end

% Create Table
rows = algs;
std_table_first_last_1000 = table(all_shapc_stds, 'VariableNames', "SHAPC Stds", 'RowNames', rows)
%std_table_first_last_1000 = table(all_shapc_vars, 'VariableNames', "SHAPC-Var", 'RowNames', rows)
mean_table = table(means, 'VariableNames', "SHAPC Means", 'RowNames', rows);
threshold_table = table(thresholds, 'VariableNames', "SHAPC Thresholds", 'RowNames', rows);
threshold = min(thresholds);
std(all_shapc_stds)
mean(all_shapc_stds)
end