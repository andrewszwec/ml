## poly regression

require(caret)

m = 4

midTermExam = c(89, 72, 94, 69)
midTermExam_2 = c(89^2, 72^2, 94^2, 69^2)
finalExam = c(96, 74, 87, 78)
df <- data.frame(cbind(midTermExam, midTermExam_2, finalExam))
df



modFit <- train(Class ~ .,method="rpart", preProcess=c("center", "scale", "pca"), data=inTrain)




hx = theta0 + theta1*x1 + theta2*x2 
# feature scaling
# divide by max-min or range of feature and mean normalisation

x1_1 = 89 
x1_1 = 94 
x = x1_1
mu = mean(midTermExam); mu
range = max(midTermExam)-min(midTermExam); range
z = (x- mu)/range; round(z,2)


# Q3
require(MASS)

m = 23
n = 5

m <- matrix(runif(23*5), ncol=5)
a <-  matrix(1,nrow=23, ncol=1)
y <- matrix(runif(23), nrow=23, ncol=1)
x <- cbind(m,a); x

# Equation for solving for regression algorithm parameters using matrices - Using the "NORMAL EQUATION"
theta = ginv(t(x) %*% x) %*% t(x) %*% y
dim(theta)


# Q4
When number of samples (m) is big and number of features (n) is big then use gradient descent instead of the normal equation and calculating the inverse matrix of ginv(t(x) %*% x) is very computationally intensive.

With n=200000 features, you will have to invert a 200001×200001 matrix to compute the normal equation. Inverting such a large matrix is computationally expensive, so gradient descent is a good choice.

# q5 
Feature scaling speeds up gradient descent by avoiding many extra iterations that are required when one or more features take on much larger values than the rest.












