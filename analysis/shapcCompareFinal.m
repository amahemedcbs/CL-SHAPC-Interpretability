% SHAPC-Mean and SHAPC-Var Comparison
algs = ["iTAML", "RPSnet", "DGR", "foster", "memo", "der"];%, "icarl"];
dataset = 'cifar100';
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

if dataset ~= "mnist"
    algs(algs=="DGR") = [];
end
%%
% Load SHAPC values (First and Last 1000)
for i=1:length(algs)
    alg = algs(i);
    save_path = sprintf("%s/%s/shapc_vals_first_last_1000.mat", alg, dataset);
    if isfile(save_path)
        shapc_struct = load(save_path);
       
        % Load SHAPC values all
        shapc_avgs = [];
        for i=1:num_sessions-1
            pair_str = 'sc' + string(i-1) +string(num_sessions-1);
            shapcs = [];
            sample_list = string(fieldnames(shapc_struct.(pair_str)));
            for k=1:length(fieldnames(shapc_struct.(pair_str)))
                sample_str = sample_list(k);
                shapcs = [shapcs; shapc_struct.(pair_str).(sample_str)];
            end
            shapc_avgs = [shapc_avgs; mean(shapcs)];

        end
        
        % Note the shapc is represented as percentage
        %disp("From averaging all tasks together:")
        shapc_mean_perc = mean(shapc_avgs);
        shapc_mean = mean(shapc_avgs) / 100;
    else
        shapc_mean_perc = NaN;
        shapc_mean = NaN;
    end
    shapc_data.(alg).first_last_1000_shapc = shapc_mean_perc;
    %shapc_data.(alg).first_last_time = shapTimes.(dataset).first_last.(alg);
end
save(shapc_path, "shapc_data")

% Create Table
%shapc_table = table(Y', 'VariableNames', ["SHAPC-Mean"], 'RowNames', X)
%abs_diff = abs(Y(1)-Y(2))
rows = algs;
columns = ["Accuracy (%)" "SHAPC-Mean (%)" "Time (hrs)"];
column_data1 = [];
column_data2 = [];
column_data3 = [];
for i=1:length(algs)
    alg = algs(i);
    column_data1 = [column_data1; shapc_data.(alg).acc];
    column_data2 = [column_data2; shapc_data.(alg).first_last_1000_shapc];
    column_data3 = [column_data3; shapc_data.(alg).first_last_1000_time];
end
first_last_1000_shapcs = column_data2;
first_last_1000_times = column_data3;
shapc_table_first_last_1000 = table(column_data1, column_data2, column_data3, ...
    'VariableNames', columns, 'RowNames', rows);
sorted_shapc_first_last_1000 = sortrows(shapc_table_first_last_1000, {'SHAPC-Mean (%)'}, {'ascend'});
disp(sorted_shapc_first_last_1000) %[output:8cdbecb6]


%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"onright","rightPanelPercent":47.2}
%---
%[output:8cdbecb6]
%   data: {"dataType":"text","outputData":{"text":"              <strong>Accuracy (%)<\/strong>    <strong>SHAPC-Mean (%)<\/strong>    <strong>Time (hrs)<\/strong>\n              <strong>____________<\/strong>    <strong>______________<\/strong>    <strong>__________<\/strong>\n\n    <strong>iTAML <\/strong>       78.66            27.186          4.4339  \n    <strong>foster<\/strong>       65.95            28.709          2.9886  \n    <strong>RPSnet<\/strong>       40.51            30.739          36.407  \n    <strong>memo  <\/strong>       68.99            31.252          5.0228  \n    <strong>der   <\/strong>       63.68            41.429          10.866  \n\n","truncated":false}}
%---
