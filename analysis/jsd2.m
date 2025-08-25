%Progress: 100%|██████████| 50/50 [29:47<00:00, 35.74s/it]

%load("iTAML\mnist\iTAML_ses_0_XA_dcts.mat")
%load("iTAML\mnist\iTAML_ses_0_XB_dcts.mat")
algorithm = 'iTAML';
dataset = 'mnist';
dataset2 = 'cifar10';

class1 = load(sprintf("%s\\%s\\%s_ses_0_XA_dcts.mat", algorithm, dataset, algorithm));
fns = fieldnames(class1);
data1 = class1.(fns{1});
p1=gkdeb(data1);

class2 = load(sprintf("%s\\%s\\%s_ses_0_XB_dcts.mat", algorithm, dataset, algorithm));
fns = fieldnames(class2);
data2 = class2.(fns{1});
p2=gkdeb(data2);

divergence = JSDiv(p1.pdf, p2.pdf);

% Assuming js_divergence_1 contains the Jensen-Shannon divergence
js_distance = sqrt(divergence);

% Display the Jensen-Shannon distance
disp(['Jensen-Shannon Distance: ', num2str(js_distance)]);
