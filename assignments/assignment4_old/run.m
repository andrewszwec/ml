%% Run Test Code

clear ; close all; clc

input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)

load('ex4data1.mat');
m = size(X, 1);

% Load the weights into variables Theta1 and Theta2
load('ex4weights.mat');

% Unroll parameters 
nn_params = [Theta1(:) ; Theta2(:)];

%% ================ Part 3: Compute Cost (Feedforward) ================
lambda = 0;
printf("computing cost...\n")

J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels, X, y, lambda);

fprintf(['Cost at parameters (loaded from ex4weights): %f '...
         '\n(this value should be about 0.287629)\n'], J);


%% =============== Part 4: Implement Regularization ===============

% fprintf('\nChecking Cost Function (w/ Regularization) ... \n')

% % Weight regularization parameter (we set this to 1 here).
% lambda = 1;

% J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
%                    num_labels, X, y, lambda);

% fprintf(['Cost at parameters (loaded from ex4weights): %f '...
%          '\n(this value should be about 0.383770)\n'], J);


%% ================ Part 5: Sigmoid Gradient  ================


% fprintf('\nEvaluating sigmoid gradient...\n')

% g = sigmoidGradient([1 -0.5 0 0.5 1]);
% fprintf('Sigmoid gradient evaluated at [1 -0.5 0 0.5 1]:\n  ');
% fprintf('%f ', g);
% fprintf('\n\n');