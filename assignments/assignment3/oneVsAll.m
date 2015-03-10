function [all_theta] = oneVsAll(X, y, num_labels, lambda)

% Some useful variables
m = size(X, 1);
n = size(X, 2);

all_theta = zeros(num_labels, n + 1);


% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================

% Set Initial theta
initial_theta = zeros(n + 1, 1);

 
% Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 50);

for c = 1:num_labels
	%% This function loops through the hand written number 0-9 and use the optimiser 
	%% to train the theta parameters for each of 402 features that identify a hand
	%% written number. 
	%% What this means is that certain features have a higher weighting/importance in 
	%% predicting a certain hand written number

	% Run fmincg to obtain the optimal theta
	% This function will return theta and the cost 
	[theta] = fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
	            initial_theta, options);
	all_theta(c,:) = theta'; 

endfor 

%% Now we have all_theta which is a matrix with one row for each of the 10
%% handwritten numbers and 402 columns representing the weighting parameter
%% for each of the features











% =========================================================================


end
