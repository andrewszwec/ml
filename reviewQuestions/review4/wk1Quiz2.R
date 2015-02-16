x = c(1,2,4,0);
y = c(0.5, 1, 2, 0);

fit <- lm(y~x)
fit


## Gradient Descent
y = theta0 + theta1*x
y = 0 + 0.5 x

theta0 = 0 - 0.1*1/3*sum(hx - y)


hx = theta0 + theta1*x

# let 
x = c(1,2,3)
y=c(1,2,3)
hx = 0 + 0.5*x
hx
m=3

theta0 = 0
theta1 = 0.5
alpha = 0.1

theta0_ = theta0 - alpha*1/m*sum(hx - y)
theta1_ = theta1 - alpha*1/m*sum((hx - y)*x) 

theta0_
theta1_


## Linear Algebra Review Questions
u = c(4,-4,-3)
v = c(4,2,4)

c(4,-4,-3)

c(16+8+16, 
-16-8-16,
-12-6-12)

16-16-12 

t(u) %*% v

a = rbind( c(1,2,3), c(2,3,4), c(5,6,7))
a

b = rbind( c(1,5,8), c(2,2,4), c(5,4,1))
b

a %*% b %*% a 
b %*% a %*% b

c = diag(3)
c

c %*% b
b %*% c

(a %*% b) %*% a
a %*% (b %*% a)



