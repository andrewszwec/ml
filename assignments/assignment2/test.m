%
%	Test file for testing ideas	
%
%
%

=
% X = [ 	1,2,3,4,5,6,7,8,9; 	2,3,4,5,6,7,8,9,10	]
%
% y = X(1,1:9).^
%
% plot(X,y, "x")




disp("hello")

%% Initialization
clear ; close all; clc


data = load('ex2data1.txt');
X = data(:, [1, 2]); y = data(:, 3);

[m, n] = size(X);
X = [ones(m, 1) X];
initial_theta = zeros(n + 1, 1);

theta = initial_theta

%---------------------------------------------------------------
%	logistic regression
%---------------------------------------------------------------

%% Cost function J(theta)
H = sigmoid( X  * theta)
S = sum( (-1 * log(H') * y) - ( (1-y)' * log(1-H) ) ) ;
J = S / m;

%% Grad - Partial derivative of d j/d theta
H = sigmoid( X  * theta)
S = sum((H - y)'* X)
grad = S / (m)

%---------------------------------------------------------------
%	fminunc
%---------------------------------------------------------------


%  Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 400);

%  Run fminunc to obtain the optimal theta
%  This function will return theta and the cost 
[theta, cost] = ...
	fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);

% Print theta to screen
fprintf('Cost at theta found by fminunc: %f\n', cost);
fprintf('theta: \n');
fprintf(' %f \n', theta);

% Plot Boundary
plotDecisionBoundary(theta, X, y);



















