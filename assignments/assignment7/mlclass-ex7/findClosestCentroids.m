function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);
m = size(X,1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%



for i = 1:m
	
	J = [];

	for k = 1:K

		% Cost function
		J(end+1) = 	sum(sum((X(i,:) .- centroids(k,:)).^2));
		 			
		
	end
	idx(i) = find(J == min(J));


% end


% for i = 1:m

% 	J = [];

% 	for k = 1:K

% 		% Cost function
% 		J(end+1) = sum(X(i,:) .- centroids(k,:))^2;
		


% 		if( k==1 )
% 			J_old = J;
% 			idx(i) = k;
% 		endif

% 		if((J < J_old) && (k>1))
% 			J_old = J;
% 			idx(i) = k;
% 		endif

		
% 	end 
% 	% d_new = sqrt((X(i,1)-centroids(1,1))^2 + (X(i,2)-centroids(1,2))^2)
% 	% d_new = sqrt((X(i,1)-centroids(2,1))^2 + (X(i,2)-centroids(2,2))^2)
% 	% d_new = sqrt((X(i,1)-centroids(3,1))^2 + (X(i,2)-centroids(3,2))^2)
% 	% J
% 	% min(J)

% end 


% Working for vectors of width 2 
% for i = 1:m
% 	X1 = X(i,1);
% 	X2 = X(i,2);

% 	for k = 1:K
% 		k1 = centroids(k,1);
% 		k2 = centroids(k,2);

% 		% Cost function
% 		% J = 1/m * sum(xi - u_ci)^2

% 		% distance formula
% 		d_new = sqrt((X1-k1)^2 + (X2-k2)^2);

% 		if(k == 1)
% 			d_old = d_new;
% 			idx(i) = k;
% 		endif 

% 		if((d_new < d_old) && (k>1) )
% 			d_old = d_new;
% 			idx(i) = k;
% 		endif
% 	end 
% end 





% =============================================================

end

