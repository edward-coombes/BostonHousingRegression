library(glmnet)

addNames <- function(d){
  
  colnames(d)[1] = "CRIM"
  colnames(d)[2] = "ZN"
  colnames(d)[3] = "INDUS"
  colnames(d)[4] = "CHAS"
  colnames(d)[5] = "NOX"
  colnames(d)[6] = "RM"
  colnames(d)[7] = "AGE"
  colnames(d)[8] = "DIS"
  colnames(d)[9] = "RAD"
  colnames(d)[10] = "TAX"
  colnames(d)[11] = "PTRATIO"
  colnames(d)[12] = "B"
  colnames(d)[13] = "LSTAT"
  colnames(d)[14] = "MEDV"
  
  return(d)
}

#get data
housingData <- read.table('~/Documents/machineLearning/project1/490-datasets/housing.asc')
housingData <- addNames(housingData)

#Set the seed for reproducibility
set.seed(123)

#prepare the model
x <- model.matrix(MEDV~.,housingData)[,-14]
y <- housingData$MEDV
lambda <- 10^seq(10,-2,length=1000)

#split the data
trainingData = sample(1:nrow(x),nrow(x)*0.7)
testData = (-trainingData)
ytest = y[testData]

#Train the linear model
formula <- MEDV ~ . 
linearModel <- lm(formula,data=housingData, subset=trainingData)
linearPredictiontest <- predict(linearModel,newdata=housingData[testData,])
linearPredictiontrain <- predict(linearModel,newdata=housingData[trainingData,])

linearRMSEtest <- sqrt(mean((linearPredictiontest-ytest)^2))
linearRMSEtrain <- sqrt(mean((linearPredictiontrain-y[trainingData])^2))


#create the cross validation object to 
crossValidation <- cv.glmnet(x[trainingData,],y[trainingData],alpha=0,lambda=lambda)

#train the ridge model
ridgeModel <- glmnet(x[trainingData,],y[trainingData], alpha = 0, lambda = lambda)

ridgePredictiontest <- predict(ridgeModel,s=crossValidation$lambda.min,newx=x[testData,])
ridgePredictiontrain <- predict(ridgeModel,s=crossValidation$lambda.min,newx=x[trainingData,])

#print(crossValidation$lambda.min)

ridgeRMSEtest <- sqrt(mean((ridgePredictiontest-ytest)^2))
ridgeRMSEtrain <- sqrt(mean((ridgePredictiontrain-y[trainingData])^2))

#Train the lasso model
lassoModel <- glmnet(x[trainingData,],y[trainingData], alpha=1,lambda=lambda)
lassoPredictiontest <- predict(lassoModel,s=crossValidation$lambda.min,newx = x[testData,])
lassoPredictiontrain <- predict(lassoModel,s=crossValidation$lambda.min,newx = x[trainingData,])

#print(crossValidation$lambda.min)

lassoRMSEtest <- sqrt(mean((lassoPredictiontest-ytest)^2))
lassoRMSEtrain <- sqrt(mean((lassoPredictiontrain-y[trainingData])^2))

#print the coefficients
#coef(linearModel)
#print(predict(ridgeModel, type = 'coefficients', s = crossValidation$lambda.min)[1:14,])
#print(predict(lassoModel, type = 'coefficients', s = crossValidation$lambda.min)[1:14,])

#compare
#print(max(housingData$MEDV))
#print(min(housingData$MEDV))

#print(linearRMSEtrain)
cat("linear RMSE",linearRMSEtest,"\n")

#print(ridgeRMSEtrain)
cat("ridge RMSE", ridgeRMSEtest,"\n")

#print(lassoRMSEtrain)
cat("lasso RMSE", lassoRMSEtest,"\n")