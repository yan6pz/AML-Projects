library(MASS)
library(ISLR)

setwd("D:/Masters/MCS-DS/Applied machine learning/Homeworks/AML-Projects/homework6")
names <- c("crim","zn","indus","chas","nox","rm","age","dis","rad","tax","ptratio","black","lstat","medv")
data <- read.table("housing.data", header=F, col.names = names)

summary(data)
head(data)

#regress only over the black variable
lr_against_black=lm(medv~black,data=data)
summary(lr_against_black)
par(mfrow=c(2,2))
plot(lr_against_black)

plot(medv~black,data)
abline(lr_against_black,col="red")



#regress over all the attributes
multi_lr=lm(medv~crim+zn+indus+chas+nox+rm+age+dis+rad+tax+ptratio+black+lstat,data)
summary(multi_lr)
plot(multi_lr)


#remove 3 outliers
data_without_outliers <- data[-c(365, 369, 373), ]

multi_lr_no_outliers=lm(medv~crim+zn+indus+chas+nox+rm+age+dis+rad+tax+ptratio+black+lstat,data_without_outliers)
summary(data_without_outliers)
plot(multi_lr_no_outliers)

