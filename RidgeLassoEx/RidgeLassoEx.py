# Autos.csv includes variables: MPG, cylinders, displacement, horsepower, weight, acceleration, origin, train
# 'train' variable indicates membership of observation in training set
# Determine best parameter for lasso and ridge models using 5 fold cross-validation
# Dependent Variable is MPG, for which the rest of the variables will be predictors   
# Finally compare Least-Squares Regression, Lasso Regression and Ridge Regression on full training data
# calculate prediction error on test data for each model


import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load data directly into dataframe df, split into train and test set
df = pd.read_csv("autos.csv")
train = df[(df["train"] == True)]
train = train.reset_index()
del train['index']


# map in folds data to perform cross validation on training set to determine best regularization parameter
train['fold'] = pd.read_csv("auto.folds",header=None)
del train["train"]
test = df[(df["train"] == False)]
del test["train"]

names = train.columns.tolist()

# X is dataframe of all independent variables
# y is single column holding dependent variable

X = train[names[1:len(names)-1]]
y = pd.DataFrame(train['mpg'])
y['fold'] = train['fold']
folds = range(5)


# We will start by finding our best Lasso Model
# Regularization parameters to choose from for our Lasso regression

lambda_ridge = np.array([0.001,0.005,0.01,0.05,0.1,0.5,1,5,10,50,100])
lambda_lasso = np.array([0.0001,0.0005,0.001,0.005,0.01,0.05,0.1])

plt.figure(1)
axis = plt.gca()
axis.set_color_cycle(2* ['b','r','g','c','k','y'])

# Initialize lists to store mean squared errors from lasso models fitted with different regularization 
# parameters and each of the mse values from the different folds in our 5 fold cross validation.
# These values will determine the optimally fit lambda value 

k = X.shape[1]
lasso_coef = np.zeros((len(lambda_lasso), k))
lasso_mse_list = []
lasso_mse_mean = []

for i,a in enumerate(lambda_lasso):
    lasso_model = Lasso(alpha=a, normalize=True)
    lasso_mse_fold = []
    for j in folds:
        X_fold_test = train[(train['fold'] == j)]
        X_fold_train = train[(train['fold'] != j)]
        y_fold_test = y[(y['fold'] == j)]
        y_fold_train= y[(y['fold'] != j)]
        del X_fold_train['mpg'],X_fold_test['mpg'], X_fold_test['fold'], X_fold_train['fold'], y_fold_train['fold'], y_fold_test['fold']
        lasso_model.fit(X_fold_train.as_matrix(), y_fold_train.as_matrix())
        mse = mean_squared_error(y_fold_test.as_matrix(), lasso_model.predict(X_fold_test.as_matrix()))
        lasso_mse_fold.append(mse)
    lasso_coef[i] = lasso_model.coef_
    lasso_mse_list.append(lasso_mse_fold)
    lasso_mse_mean.append(np.array(lasso_mse_list).mean())

# best lasso parameter

min_index_lasso = lasso_mse_mean.index(min(lasso_mse_mean))
print "Best Lasso: ", lambda_lasso[min_index_lasso]

# plot lasso coefficients

for coef in lasso_coef.T:
    plt.plot(lambda_lasso, coef)

plt.xlabel('Lambda')
plt.ylabel('Coefficients')
plt.xlim(min(lambda_lasso),max(lambda_lasso))
plt.title('Lasso')

# plots of average lasso error across folds. Includes red line for best lambda

plt.figure(2)
plt.plot(-np.log(lambda_lasso),
    np.sqrt(np.array(lasso_mse_list)).mean(axis=1))
plt.axvline(-np.log(lambda_lasso[min_index_lasso]), color = 'r')
plt.xlabel(r'-log(lambda)')
plt.ylabel('RMSE')
plt.title('Lasso')


# Now we fit our best Ridge model, similar to above steps.

plt.figure(3)
axis = plt.gca()
axis.set_color_cycle(2* ['y','r','g','c','k','b'])


ridge_coef = np.zeros((len(lambda_ridge), k))
ridge_mse_list = []
ridge_mse_mean = []

for i,a in enumerate(lambda_ridge):
    ridge_model = Ridge(alpha=a, normalize=True)
    ridge_mse_fold = []
    for j in folds:
        X_fold_test = train[(train['fold'] == j)]
        X_fold_train = train[(train['fold'] != j)]
        y_fold_test = y[(y['fold'] == j)]
        y_fold_train= y[(y['fold'] != j)]
        del X_fold_train['mpg'],X_fold_test['mpg'], X_fold_test['fold'], X_fold_train['fold'], y_fold_train['fold'], y_fold_test['fold']
        ridge_model.fit(X_fold_train.as_matrix(), y_fold_train.as_matrix())
        mse = mean_squared_error(y_fold_test.as_matrix(), ridge_model.predict(X_fold_test.as_matrix()))
        ridge_mse_fold.append(mse)
    ridge_coef[i] = ridge_model.coef_[0]
    ridge_mse_list.append(ridge_mse_fold)
    ridge_mse_mean.append(np.array(ridge_mse_list).mean())


# Best Ridge Regularization parameter

min_index_ridge = ridge_mse_mean.index(min(ridge_mse_mean))
print "Best Ridge: ", lambda_ridge[min_index_ridge]  

for coef in ridge_coef.T:
    plt.plot(lambda_ridge, coef)

plt.xlabel('Regularization')
plt.ylabel('Coefficients')
plt.xlim(min(lambda_ridge),max(lambda_ridge))
plt.title('Ridge')

plt.figure(4)
plt.plot(-np.log(lambda_ridge),
    np.sqrt(np.array(ridge_mse_list)).mean(axis=1))
plt.axvline(-np.log(lambda_ridge[min_index_ridge]), color = 'red')
plt.xlabel(r'-log(lambda)')
plt.ylabel('RMSE')
plt.title('CV Error')

plt.show()


# Using best reg. parameters from above to find prediction error on our test set

df2 = pd.read_csv("autos.csv")

train2 = df2[(df2["train"] == True)]
del train2["train"]
test2 = df2[(df2["train"] == False)]
del test2["train"]

names = train2.columns.tolist()

X_train = train2[names[1:len(names)]]
X_test = test2[names[1:len(names)]]
y_train = train2[names[0]]
y_test = test2[names[0]]

train = np.matrix(X_train)
test = np.matrix(X_test)

# Building lasso, ridge and basic linear model to compare results

lasso_model2 = Lasso(alpha=0.01, normalize=True)
ridge_model2= Ridge(alpha=0.01, normalize=True)
lm = LinearRegression()

lasso_model2.fit(X_train.as_matrix(), y_train.as_matrix())
ridge_model2.fit(X_train.as_matrix(), y_train.as_matrix())
lm.fit(train, y_train.as_matrix())

lasso_model2.predict(X_test.as_matrix())
ridge_model2.predict(X_test.as_matrix())
lm.predict(test)

mse_lasso = mean_squared_error(y_test.as_matrix(), lasso_model2.predict(X_test.as_matrix()))
mse_ridge = mean_squared_error(y_test.as_matrix(), ridge_model2.predict(X_test.as_matrix()))
mse_lm= mean_squared_error(y_test.as_matrix(), lm.predict(test))

# MSE for each model

print "Lasso MSE: ", mse_lasso
print "Ridge MSE: ", mse_ridge
print "Least Squares MSE: ", mse_lm
print 'Coefficients:'


