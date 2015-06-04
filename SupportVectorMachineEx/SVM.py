# Support Vector Regression Example using scikit learn package
# train and test data included in folder come from auto-mpg dataset from UCI repository
# outliers have been artificially added to show power of SVR
# while training RMSE will be higher, result should show SVR's ability to avoid overfitting


import numpy as np
from sklearn.svm import SVR
from sklearn import preprocessing
from sklearn import linear_model
 
train_data = np.genfromtxt('data.train', delimiter = ',', skip_header = 1)
train_data_scale = preprocessing.scale(train_data)
test_data = np.genfromtxt('data.test', delimiter = ',', skip_header = 1)
test_data_scale = preprocessing.scale(test_data)
train_y = train_data[:,0]
train_x = train_data_scale[:,1:]
test_y = test_data[:,0]
test_x = test_data_scale[:,1:]
 
svr_model = SVR()
svr_model.fit(train_x, train_y)

# Using SVR model to determine prediction accuracy on our training set

train_predict = svr_model.predict(train_x)
train_mse = np.mean((train_predict - train_y)**2)
train_rmse = np.sqrt(train_mse)

# Using SVR model to determine prediction accuracy on our test set
test_predict = svr_model.predict(test_x)
test_mse = np.mean((test_predict - test_y)**2)
test_rmse = np.sqrt(test_mse)
 
# Least squares regression models for comparison

lsr = linear_model.LinearRegression()
lsr.fit(train_x, train_y)


# RMSE for linear models
linear_train_predict = lsr.predict(train_x)
linear_train_mse = np.mean((linear_train_predict - train_y)**2)
linear_train_rmse = np.sqrt(linear_train_mse)

linear_test_predict = lsr.predict(test_x)
linear_test_mse = np.mean((linear_test_predict - test_y)**2)
linear_test_rmse = np.sqrt(linear_test_mse)


print 'SVR train result: ' + str(train_rmse) 
print 'SVR test result: ' + str(test_rmse)
print 'Linear model train results: ' + str(linear_train_rmse) 
print 'Linear model test results: '+ str(linear_test_rmse)