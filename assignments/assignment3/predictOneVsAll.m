function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

m = size(X, 1);
num_labels = size(all_theta, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters (one-vs-all).
%               You should set p to a vector of predictions (from 1 to
%               num_labels).
%
% Hint: This code can be done all vectorized using the max function.
%       In particular, the max function can also return the index of the 
%       max element, for more information see 'help max'. If your examples 
%       are in rows, then, you can use max(A, [], 2) to obtain the max 
%       for each row.
%       

% X looks like this
% 
% 			feature1 	feature2 	feature3 .... feature 401
% sample 1	1			33			22
% sample 2 	1			55			66	
% sample 3 	1			33			32
% sample 4 	1			88			77
% ...
% sample 5000
% 
% 
% all_theta looks like this
% 			
% 			parameter 1 	parameter 2 ... 	parameter 402
% label 1 	100				199			... 	277
% label 2 	233				77			...		88	
% ....	
% label 10 	865				665			...		445

% for all samples in X loop through and apply the formula
for i=1:m,
    [x, p(i)] = max(sigmoid(all_theta * X(i,:)'));
end

%% This thing calculates the likelihood of the hand written text being each of 0-9 
%% for all 5000 samples. It then finds the maximum likelihood using the max function 
%% and returns the column the max occurs in. The columns are in order and correspond 
%% the numbers 0-9 (note here 1-9,0 are mapped to 1-10)


% =========================================================================


end
