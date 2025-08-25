%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function [p_value,CI_class1,CI_class2] 
% = anova1_class1_class2
% The anova1 with multcompare functon performs a 1 way anova 
% Output
% p_value
% CI_class1: confidence interval of class 1
% CI_class2: confidence interval of class 2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [p_value,CI_class1,CI_class2] = anova1_class1_class2(class1, class2)

% Load data from file and transpose data
%load inclass.mat
%load outofclass.mat
%{
load("iTAML\mnist\iTAML_ses_0_XA_dcts.mat")
load("iTAML\mnist\iTAML_ses_0_XA_dcts.mat")
load("iTAML\mnist\iTAML_ses_0_XB_dcts.mat")
load("iTAML\mnist\iTAML_ses_0_XB_dcts.mat")
%}
%{
load("RPSnet\cifar10\RPSnet_ses_0_XA_dcts.mat")
load("RPSnet\cifar10\RPSnet_ses_0_XB_dcts.mat")
%}

data = [class1 class2];
group = [ones(1,length(class1)) 2*ones(1,length(class2))];

% Perform 1 way ANOVA
[p_value,~,stats] = anova1(data,group, "off");

% Calculate 95% confidence intervals for classes 1 and 2
% Find standard errors
x = class1;
m_class1 = mean(x);
SEM_class1 = std(x)/sqrt(length(x));               
x = class2;
m_class2 = mean(x);
SEM_class2 = std(x)/sqrt(length(x));               

% Establish z-score for 95% confidence interval
zs = 1.96;                                  

% Calculate 95% confidence intervals
CI_class1 = [m_class1 - zs*SEM_class1, m_class1 + zs*SEM_class1];  
CI_class2 = [m_class2 - zs*SEM_class2, m_class2 + zs*SEM_class2];  

% Print confidence intervals
fprintf('Class 1 95%% CI: [%f, %f]\n', CI_class1(1), CI_class1(2));
fprintf('Class 2 95%% CI: [%f, %f]\n', CI_class2(1), CI_class2(2));

% Define the means and confidence intervals
means = [mean(class1), mean(class2)];
CIs = [CI_class1; CI_class2];

% Plotting confidence intervals as boxes (Figure 2)
figure;

% Colors for each model
colors = {'k', 'k'};  

% Plot the boxes for confidence intervals
hold on;
for i = 1:2
    rectangle('Position', [i-0.25, CIs(i,1), 0.5, CIs(i,2) - CIs(i,1)], 'EdgeColor', colors{i}, 'LineWidth', 2);
end

% Add mean markers
for i = 1:2
    plot(i, means(i), 's', 'MarkerEdgeColor', colors{i}, 'MarkerFaceColor', colors{i});
end

% Customize the plot
set(gca, 'XTick', [1 2], 'XTickLabel', {'Class 1', 'Class 2'});
xlabel('Classes', 'FontWeight', 'bold');
ylabel('DCT norm', 'FontWeight', 'bold');

% Add grid lines for clarity
grid on;

hold off;
