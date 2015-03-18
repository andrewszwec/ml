function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%


% Calculate the best C and sigma

% Steps
% 1. 	Train the SVM using your choice of C and sigma on X and y
%		Examples of C and Sigma are [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
%		Build 64 models
% 2.	Now apply the trained model to the cross validation set
%		and calculate the error on the CV for each of the 64 models.
%		Choose the model with the lowest cross validation error


mySigma = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
myC = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
details = [];

for s = 1:length(mySigma)
	sigma = mySigma(s)

	for c = 1:length(myC)
		C = myC(c)

		% Step 1
		model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));

		% just save C, sigma, error (this is a standard)

		predictions = svmPredict(model, Xval);
		error_cv = mean(double(predictions ~= yval));

		details(end+1, :) = [C, sigma, error_cv];

	end
end

% Get the minimum error in the error col
min_error_cv = min( details(:,3) );
% make a bit mask that tells you where the min value is
idx = ( details(:,3) == min_error_cv );

% store the entire row in a vector
C_sigma_err = details(idx,:);

% set C and sigma
C = C_sigma_err(1);
sigma = C_sigma_err(2);

% =========================================================================

end
