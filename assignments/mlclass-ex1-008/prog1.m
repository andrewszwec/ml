// Programming Assignment 1

// set working directory
cd /Users/andrewszwec/Documents/Coursera/ml/assignments/mlclass-ex1-008/mlclass-ex1


// Make plots use x11
setenv GNUTERM x11

// This is some code snippets to run to load your coding environment
fprintf('Plotting Data ...\n')
data = load('x_data.txt');
X = data(:, 1:4); y = data(:, 5);
m = length(y); % number of training examples

X = [ones(m, 1), data(:,1:4)]; % Add a column of ones to x
theta = zeros(6, 1); % initialize fitting parameters

% Some gradient descent settings
iterations = 1500;
alpha = 0.01;

% compute and display initial cost

H = (theta'*X')';
S = sum((H - y) .^ 2); %'
J = S / (2*m);
fprintf('J is the cost = %d \r\n', J );


% X with ones is 6 x 5
% theta is 5 x 1


% run gradient descent
theta = gradientDescent(X, y, theta, alpha, iterations);