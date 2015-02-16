X = [1,2,3,4,5;
6,7,8,9,10;
11,12,13,14,15;
16,17,18,19,20;
1,2,3,4,5;
6,7,8,9,10
]

mu = mean(X);
sigma = std(X);

for i = 1 : length(X)
	%% for each column do something
	for j=1 : size(X)(2)
		x_norm(i,j) = ( X(i,j) - mu(j)) / sigma(j);
	end

end


disp(x_norm)
