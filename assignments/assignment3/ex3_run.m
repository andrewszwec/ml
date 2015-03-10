
clear ; close all; clc

input_layer_size  = 400;  % 20x20 Input Images of Digits
num_labels = 10;          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)
load('ex3data1.mat'); % training data stored in arrays X, y
m = size(X, 1);

% Randomly select 100 data points to display
rand_indices = randperm(m);
sel = X(rand_indices(1:100), :);

% displayData(sel);

lambda = 0.1;
% [all_theta] = oneVsAll(X, y, num_labels, lambda);

% Set Initial theta
% Some useful variables
m = size(X, 1);
n = size(X, 2);
initial_theta = zeros(n + 1, 1);
% Add ones to the X data matrix
%X = [ones(m, 1) X];

%label = 1;
%lrCostFunction(initial_theta, X, (y == label), lambda);

[all_theta] = oneVsAll(X, y, num_labels, lambda);


%% ================ Part 3: Predict for One-Vs-All ================
%  After ...
pred = predictOneVsAll(all_theta, X);

printf('\nTraining Set Accuracy: %f\n\n', mean(double(pred == y)) * 100);

% debugging
% Add ones to the X data matrix
X = [ones(m, 1) X];

[r, p1] = max(sigmoid(all_theta * X(1,:)'))

m = size(X, 1);
out = zeros (1, 10);
for i=1:m,
   out(i,:) = (sigmoid(all_theta * X(i,:)'));
end

out(1:5, :)







