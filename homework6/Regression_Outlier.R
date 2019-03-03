# Title     : Outlier detection using linear regression
# Objective : To detect and train a linear regression model against housing data
# Created by: minyuan gu
# Created on: 2/03/2019
library(MASS)
col_names <- c("crim","zn","indus","chas","nox","rm","age","dis","rad","tax","ptratio","b","lstat","medv")
inputdata <- read.table("./homework6/housing.data", header=F, col.names = col_names)
cooksD_lv <- c(0.1, 0.2, 0.5, 1.0)

printStats <- function(model){
    sm <- summary(model)
    print("Model: ")
    print(sm$call)
    print(paste("R squared:", sm$r.squared))
    print("Top 5 standardized residuals: ")
    print(head(sort(rstandard(model), decreasing=TRUE),5))
    print("Top 5 leverage: ")
    print(head(sort(hatvalues(model), decreasing=TRUE),5))
    print("Top 5 Cook's Distance: ")
    print(head(sort(cooks.distance(model), decreasing=TRUE),5))
}
#########################################
#       Model #1 - original data        #
#########################################
fit1<-lm(medv ~ . -medv, data = inputdata)
printStats(fit1)
plot(fit1, cook.levels= cooksD_lv)

#########################################
#   Model #2 - outliers removed data    #
#########################################
# now remove the outliers of point 369, 372 & 373 - due to high standardized residuals and high leverage and high cook's distance
removed <- inputdata[-c(369, 372, 373),]
fit2<-lm(medv ~ . - medv, data = removed)
printStats(fit2)
plot(fit2, cook.levels= cooksD_lv)
# also removed  366, 368, 370, 371 which have large standardized residual & cook's distance
# and we repeated the above procedure.
# we remove total 365, 366, 368, 369, 370, 371, 372, 373, 375, 413
removed <- inputdata[-c(365, 366, 368, 369, 370, 371, 372, 373, 375, 413),]
fit2<-lm(medv ~ . - medv, data = removed)
printStats(fit2)
plot(fit2, cook.levels= cooksD_lv)

#removed <- inputdata[c(-slst$ix[1],-slst$ix[2], -365, -413, -366, -368, -370, -371, -372, -375),]
###########################################################
#   Model #3 - apply boxcox with outliers removed data    #
###########################################################
results <- boxcox(fit2)
lambda <- results$x[which.max(results$y)]
print("Suggested lambda from boxcox: ")
print(lambda)
# now calculate using lambda transfored Y
transformed <- removed
transformed$medv<- ((transformed$medv)^lambda - 1) / lambda
fit3<- lm(medv ~ . - medv, data = transformed)
printStats(fit3)
plot(fit3, cook.levels= cooksD_lv)

# now print the predicted data against true data
predicted = (predict(fit3, removed[,-14])*lambda+1)^(1/lambda)
plot(removed[,14], predicted, xlab="True House Price",ylab="Fitted House Price",main='True House Price vs Fitted Price - outlier removed')

predicted_all = (predict(fit3, inputdata[,-14])*lambda+1)^(1/lambda)
plot(inputdata[,14], predicted_all, xlab="True House Price",ylab="Fitted House Price",main='True House Price vs Fitted Price - all data')
