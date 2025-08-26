algorithm = 'iTAML'
dataset = 'mnist'
dataset2 = 'cifar10'

class1 = load(sprintf("%s\\%s\\%s_ses_0_XA_dcts.mat", algorithm, dataset, algorithm));
fns = fieldnames(class1);
data1 = class1.(fns{1});

class2 = load(sprintf("%s\\%s\\%s_ses_4_XA_dcts.mat", algorithm, dataset, algorithm));
fns = fieldnames(class2);
data2 = class2.(fns{1});

% Flatten the data to vectors
data1 = data1(:); % Reshape to a column vector
data2 = data2(:); % Reshape to a column vector

% Find the combined range
min_x = min([min(data1), min(data2)]);
max_x = max([max(data1), max(data2)]);

% Choose a resolution (number of points)
num_points = 100; % Or any suitable number

% Create the common x vector
common_x = linspace(min_x, max_x, num_points);

%clear p_options;
p_options.x = common_x;

% Estimate PDF 1
p1 = gkdeb(data1, p_options);
pdf1 = p1.pdf;
size(pdf1) % Should be 1 x 100 (or the length of x1)

% Estimate PDF 2
p2 = gkdeb(data2, p_options);
pdf2 = p2.pdf;
size(pdf2) % Should be 1 x 100 (or the length of x2)

% 3. Handle potential negative values
pdf1(pdf1 < 0) = 0;
pdf2(pdf2 < 0) = 0;

% Calculate Jensen-Shannon Divergence
js_divergence_1 = JSDiv(pdf1, pdf2);
disp(['Jensen-Shannon Divergence (JSDiv): ', num2str(js_divergence_1)]);