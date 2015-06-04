## Implementation of Stochastic Gradient Descent, without R library, to estimate the parameters of a Ridge Regression
## We use 100 epochs and learning rates of 0.000025, 0.00055 and 0.0075 to find best model
## dataset is derived from https://archive.ics.uci.edu/ml/datasets/Forest+Fires
## data is already partitioned into forestfire-train.csv  and forestfire-test.csv


train = read.csv('forestfire-train.csv')
test = read.csv('forestfire-test.csv')
library(ggplot2)

getRMSE <- function(pred, actual) {
  error=sqrt(mean((pred-actual)^2))
  return(error)
}

addIntercept <- function(mat) {
  ## add intercept to the matrix
  allones= rep(1, nrow(mat))
  return(cbind(Intercept=allones, mat))
}

predictSamples <- function(beta, mat) {
    ## TODO: compute the predicted value using matrix multiplication
    ## Note that for a single row of mat, pred = sum_i (beta_i * feature_i)
    return (mat %*% beta)
}

MAX_EPOCH = 100

## Build sgd function

sgd <- function(learn.rate, lambda, train, test, epoch=MAX_EPOCH) {
  ## convert the train and test to matrix format
  train.mat = as.matrix(train) 
  test.mat = as.matrix(test)

  N = nrow(train.mat)
  d = ncol(train.mat)

  ## standardize the columns of both matrices
  for (i in 1:(d-1)){
    train.mat[,i]=scale(train.mat[,i])
    test.mat[,i]=scale(test.mat[,i])
  }
  
  tmat <- addIntercept(train.mat[, -d])
  testmat <- addIntercept(test.mat[, -d])

  beta = rep(0.5,d)
  j = 1
  
  # initialize dataframe to store MSE from our training set
  mse.df <- NULL
  
  # predict training residuals
  pred_train =predictSamples(beta, tmat)
  pred_test = predictSamples(beta, testmat)
  tMse = getRMSE(pred_train, train$area)
  testMSE = getRMSE(pred_test, test$area)
  mse.df <- rbind(mse.df, data.frame(epoch=j, train=tMse, test=testMSE))

  # Make 100 passes through training data 
  
  while(j < MAX_EPOCH){  
    j=j+1;
    # for each row in the training data
    for (n in seq(1:N)){
      beta_transpose= t(beta)-learn.rate*((tmat[n,] %*% beta-train.mat[n,d])%*%tmat[n,])
      beta=t(beta_transpose)
    }
    pred_train = predictSamples(beta, tmat)
    pred_test = predictSamples(beta, testmat)
    tmp_test <- data.frame(pred=pred_test, actual=test$area, type="test")
    tmp_train <- data.frame(pred=pred_train, actual=train$area, type="train")
    tmp <- rbind(tmp_train, tmp_test)
    ggplot(tmp, aes(x=pred, y=actual, color=type)) + theme_bw() + geom_point()

    tMse = getRMSE(pred_train, train$area)
    testMSE = getRMSE(pred_test, test$area)
    mse.df <- rbind(mse.df, data.frame(epoch=j, train=tMse, test=testMSE))
  } 
  return(mse.df)
}

## Plot RMSE vs Epochs to see where our error is minimized for each learning rate

results_0.0075 <- sgd(.0075, .1, train, test, epoch=MAX_EPOCH)
qplot(epoch, test, data = results_0.0075)
results_0.000025 <- sgd(.000025, .1, train, test, epoch=MAX_EPOCH)
qplot(epoch, test, data = results_0.000025)
results_0.00055 <- sgd(.00055, .1, train, test, epoch=MAX_EPOCH)
qplot(epoch, test, data = results_0.00055)

## Minimum RMSE found for each of the learning rates as we continue to make passes through our training data 

which.min(results_0.0075$test)
which.min(results_0.00055$test)
which.min(results_0.000025$test)



  