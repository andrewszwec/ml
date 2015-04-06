function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Add ones to the X data matrix
X = [ones(m, 1) X];

%------------------------------------------------------------------------------
%% FROM ASSIGNMENT 3

% H = sigmoid( X  * theta);
% temp = theta; 
% temp(1) = 0;   % because we don't add anything for j = 0 ' 

% %% The secret was to use element wise matrix multiplication instead of transpose matrices
% J = (1/m)*sum(-y.*log(H) .-(1.-y).*log(1.-H)) + ((lambda/(2*m))*sum(temp.^2));

% % gradient
% grad = 1/m * (X'*(H .- y));
% grad = grad + 1/m .* (lambda * temp ) ;

%------------------------------------------------------------------------------
%% DO SOMETHING USEFUL

printf("|\t making z2\t|\n");
z_2 = X * Theta1' ;

printf("|\t making a2\t|\n");
a_2 = sigmoid(z_2);

printf("|\t adding ones\t|\n");
% Add ones to the X data matrix
a_2 = [ones(size(a_2,1), 1) a_2];

printf("|\t making z3\t|\n");
z_3 =  a_2 * Theta2';

printf("|\t making H\t|\n");
H = sigmoid(z_3);

printf("|\t size of H \t|\n");
size(H)


temp1 = Theta1; 
temp1(1) = 0;   % because we don't add anything for j = 0 ' 

temp2 = Theta2; 
temp2(1) = 0; 

% With regularisation
S = sum(-y'*log(H) .-(1.-y)'*log(1.-H));

T1 = sum(temp1.^2);
T2 = sum(temp2.^2);

J = (1/m)* S  %+ lambda/(2*m) * sum((T1 + T2));



%------------------------------------------------------------------------------
s = 0;
for i=1:m,
    x_i = [1 X(i,:)]';
    
    y_i = zeros(num_labels, 1);
    y_i(y(i)) = 1;
    
    z_2 = Theta1 * x_i;
    a_2 = [1; sigmoid(z_2)];
    
    z_3 = Theta2 * a_2;
    h = sigmoid(z_3);
    
    d_3 = h - y_i;
    d_2 = Theta2' * d_3;
    d_2 = d_2(2:end) .* sigmoidGradient(z_2);
    
    Theta2_grad = Theta2_grad  (d_3 * a_2');
    Theta1_grad = Theta1_grad  (d_2 * x_i');
    
    s = s  sum(-y_i .* log(h) .- (1.-y_i) .* log(1.-h));
end

t_1 = Theta1(:,2:end);
t_2 = Theta2(:,2:end);

Theta1_grad = [Theta1_grad(:,1)/m ((Theta1_grad(:,2:end)/m)  (lambda/m) * t_1)];
Theta2_grad = [Theta2_grad(:,1)/m ((Theta2_grad(:,2:end)/m)  (lambda/m) * t_2)];

t_1 = sum(sum(t_1.^2));
t_2 = sum(sum(t_2.^2));
r = (lambda/(2*m))*(t_1  t_2);

J = (1/m) * s  r;



%------------------------------------------------------------------------------
%% GRAVEYARD


% UNCOMMENT
% mytemp  = zeros(m, size(X,2));
% for i = 1 : m
% 	for k = 1 : input_layer_size
% 		mytemp(i,:) = X(i,:) .* Theta1(k,:)
% 	end
% end
% END UNCOMMENT





% FROM ASSIGNMENT 3
% --------------------------
%H = sigmoid( X .* Theta1); 
% temp = Theta1; 
% temp(1) = 0;   % because we don't add anything for j = 0 ' 

%% With regularisation
% J = (1/m)*sum(-y.*log(H) .-(1.-y).*log(1.-H)) + ((lambda/(2*m))*sum(temp.^2));

%% LECTURE NOTES
% --------------------------
% z(2) = THETA(1).X or renaming = THETA(1).a(1)
% a(2) = g(z(2)) = sigmoid(z(2)) element wise

% x0 and a0 are bias unit = 1 that are not written

% z(3) = THETA(2) . a(2)
% h(x) = a(3) = g(z(3))

% Back Propogation 
% Δ(2)ij:=Δ(2)ij+δ(3)i∗(a(2))j
% BACK PROPOGATION VECTORISED
% Δ(2):=Δ(2)+δ(3)∗(a(2))T

%------------------------------------------------------------------------------
%% NEED THIS FOR GRADIENT !!!

% printf("making z2\n");
% z_2 = Theta1 .* X;

% printf("making a2\n");
% a_2 = sigmoid(z_2);

% printf("adding ones\n");
% % Add ones to the X data matrix
% a_2 = [ones(size(a_2,1), 1) a_2];

% printf("making z3\n");
% z_3 = Theta_2 .* a_2;

% printf("making hx\n");
% hx = sigmoid(z_3);

% printf("size of hx\n");
% size(hx)
%------------------------------------------------------------------------------
%% CODE FROM QUIZ

% a2 = zeros (3, 1);
% for i = 1:3
%   for j = 1:3
%     a2(i) = a2(i) + x(j) * Theta1(i, j);
%   end
%   a2(i) = sigmoid (a2(i));
% end
%------------------------------------------------------------------------------








% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
