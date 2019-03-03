library(MASS)

#setwd("~/Desktop/AML/AML-Projects/homework6")
names <- c("crim","zn","indus","chas","nox","rm","age","dis","rad","tax","ptratio","black","lstat","medv")
data <- read.table("./homework6/housing.data", header=F, col.names = names)

summary(data)
head(data)

estimateStatistics <- function(regression_model){
  #1 Estimate R2
  rsq<-summary(regression_model)$r.squared
  print("R-squared: ")
  print(rsq)
  #2 Estimate residuals
  residuals <- residuals(regression_model)
  print("Residuals: ")
  print(residuals)
  #3 Estimate the hat matrix leverage
  leverage <- hatvalues(regression_model)
  print("Leverage: ")
  print(leverage)
  #4 Estimate cook distance
  # identify D values > 4/(n-k-1)
  cutoff <- 4/((nrow(data)-length(regression_model$coefficients)-2))
  print("Cutoff: ")
  print(cutoff)
  plot(regression_model, which=4, cook.levels=cutoff)
  # Influence Plot
  #influencePlot(regression_model, id.method="identify", main="Influence Plot", sub="Circle size is proportial to Cook's Distance" )

  return(T)
}

plot_summarize <- function(regression_model){
  summary(regression_model)
  par(mfrow=c(2,2))
  plot(regression_model)
  par(mfrow=c(1,1))
}

regress <- function(data){
  multi_lr=lm(medv~crim+zn+indus+chas+nox+rm+age+dis+rad+tax+ptratio+black+lstat,data)
  estimateStatistics(multi_lr)

  #resulting model
  plot_summarize(multi_lr)

  return(multi_lr)
}


#regress over all the attributes
regress(data)

#remove top 3 influential points
data_without_3_outliers <- data[-c(365, 369, 373), ]

regress(data_without_3_outliers)

#remove top 6 influential points
data_without_6_outliers <- data[-c(366, 368, 370,365, 369, 373), ]

regress(data_without_6_outliers)

#remove top 10 influential points
data_without_10_outliers <- data[-c(366, 368, 370,365, 369, 373,372,371,381,413), ]

model_cleaned <- regress(data_without_10_outliers)

#Box-Cox transformation choosing lambda
bc<-boxcox(model_cleaned,lambda=seq(0,1,by=.1))
best_lambda <- bc$x[which.max(bc$y)]
print("Best Lambda: ")
print(best_lambda)

#transform the model based on box-cox's lambda
transformed_model <- lm(((medv^best_lambda-1)/best_lambda) ~ crim+zn+indus+chas+nox+rm+age+dis+rad+tax+ptratio+black+lstat,data_without_10_outliers)

#Standardized Residuals vs the Fitted values
st_residuals <- rstandard(transformed_model)
plot(st_residuals,data_without_10_outliers$medv,
     xlab="Standardized Residuals",ylab="Fitted values",main='Standardized Residuals vs the Fitted values')

#Predicted vs Actual values plot
plot((predict(transformed_model)*best_lambda+1)^(1/best_lambda),data_without_10_outliers$medv,
     xlab="predicted",ylab="actual", main="Predicted vs Actual values")
