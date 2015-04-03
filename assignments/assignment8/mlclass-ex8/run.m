

%% Initialization
clear ; close all; clc

load('ex8data1.mat');

%  Visualize the example dataset
% plot(X(:, 1), X(:, 2), 'bx');
% axis([0 30 0 30]);
% xlabel('Latency (ms)');
% ylabel('Throughput (mb/s)');

%  Estimate my and sigma2
[mu sigma2] = estimateGaussian(X);


pval = multivariateGaussian(Xval, mu, sigma2);

[epsilon F1] = selectThreshold(yval, pval);
fprintf('Best epsilon found using cross-validation: %e\n', epsilon);
fprintf('Best F1 on Cross Validation Set:  %f\n', F1);
fprintf('   (you should see a value epsilon of about 8.99e-05)\n\n');

%  Find the outliers in the training set and plot the
outliers = find(p < epsilon);

%  Draw a red circle around those outliers
hold on
plot(X(outliers, 1), X(outliers, 2), 'ro', 'LineWidth', 2, 'MarkerSize', 10);
hold off
