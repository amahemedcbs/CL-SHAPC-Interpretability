algs = ["iTAML", "RPSnet", "DGR", "foster", "memo", "der", "icarl", "dsal"];
dataset = 'cifar10' %[output:80322494]
if dataset == "cifar100"
    num_sessions = 10;
else
    num_sessions = 5;
end

if dataset ~= "mnist"
    algs(algs=="DGR") = [];
end
%{
if dataset ~= "cifar10"
    algs(algs=="icarl") = [];
    algs(algs=="dsal") = [];
end
%}
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
%%
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
%%
% Create Table
rows = algs;
std_table = table(all_shapc_stds, 'VariableNames', "SHAPC Stds", 'RowNames', rows) %[output:40ef5dc9]
fprintf("Std of Stds: %f", std(all_shapc_stds)); %[output:052cc99a]
fprintf("Avg of Stds: %f", mean(all_shapc_stds)); %[output:32e2ea31]
mean_table = table(means, 'VariableNames', "SHAPC Means", 'RowNames', rows);
threshold_table = table(thresholds, 'VariableNames', "SHAPC Thresholds", 'RowNames', rows);

%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"onright"}
%---
%[output:80322494]
%   data: {"dataType":"textualVariable","outputData":{"name":"dataset","value":"'cifar10'"}}
%---
%[output:40ef5dc9]
%   data: {"dataType":"tabular","outputData":{"columnNames":["SHAPC Stds"],"columns":1,"dataTypes":["single"],"header":"7×1 table","name":"std_table","rowNames":["iTAML","RPSnet","foster","memo","der","icarl","dsal"],"rows":7,"type":"table","value":[["4.9120"],["3.2777"],["4.5173"],["4.3782"],["4.5347"],["4.7045"],["5.4388"]]}}
%---
%[output:052cc99a]
%   data: {"dataType":"text","outputData":{"text":"Std of Stds: 0.657452","truncated":false}}
%---
%[output:32e2ea31]
%   data: {"dataType":"text","outputData":{"text":"Avg of Stds: 4.537583","truncated":false}}
%---
