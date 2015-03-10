function [J, grad] = lrCostFunction(theta, X, y, lambda)

% Initialize some useful values
m = length(y); % number of training examples
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================


H = sigmoid( X  * theta);
temp = theta; 
temp(1) = 0;   % because we don't add anything for j = 0 ' 

%% The secret was to use element wise matrix multiplication instead of transpose matrices
J = (1/m)*sum(-y.*log(H) .-(1.-y).*log(1.-H)) + ((lambda/(2*m))*sum(temp.^2));

% gradient
grad = 1/m * (X'*(H .- y));
grad = grad + 1/m .* (lambda * temp ) ;



% =============================================================

grad = grad(:);

end
