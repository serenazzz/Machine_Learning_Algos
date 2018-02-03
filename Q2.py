'''
Team member:
Brian Allen (ba2542)
Haozheng Ni (hn2318)
Serena Zhang (mz2642)
'''

#---------Set up --------
from scipy.io import loadmat
import os
import numpy as np
import numpy.linalg as npla


# Import wine data
wine = loadmat('wine.mat')
traindata = wine['data']
trainlabels = wine['labels'].astype('float')
testdata = wine['testdata']
testlabels = wine['testlabels'].astype('float')

# Store variable names
variables = ['intercept',
             'fixed acidity',
             'volatile acidity',
             'citric acid',
             'residual sugar',
             'chlorides',
             'free sulfur dioxide',
             'total sulfur dioxide',
             'density',
             'pH',
             'sulphates',
             'alcohol']

# Store number of variables
numVariables = traindata.shape[1]


#---------Ordinary Least Squares--------
OLS_betas = npla.inv(traindata.T @ traindata) @ traindata.T @ trainlabels
OLS_yhat = testdata @ OLS_betas
OLS_MSE = (OLS_yhat.T @ OLS_yhat - 2*(OLS_yhat.T @ testlabels) + testlabels.T @ testlabels)/len(OLS_yhat)


# Solution to 2a: test risk of the ordinary least squares estimator

print("Test squared loss risk is {:.5f}.".format(OLS_MSE.item()))


#---------Sparse Linear Predictor--------

'''
Initialize variables to store values associated with minimum MSE
'''

SLP_minMSE = float("inf")
SLP_minMSEIndex = [0,1,2,3]
SLP_minMSEBetas = []

'''
loop over all combination of three variables from d=1,...,12 and calculate the MSE on the training data
keep the combination with the smallest train error
'''
for i in range(1,numVariables):
    for j in range(i+1,numVariables):
        for k in range(j+1,numVariables):

            # Define training data and betas
            x = traindata[:,[0,i,j,k]]
            y = trainlabels
            SLP_betas = npla.inv(x.T @ x) @ x.T @ y

            # Define test data and predictions
            SLP_yhat = x @ SLP_betas

            # Calculate MSE
            tmpMSE = (SLP_yhat.T @ SLP_yhat - 2*(SLP_yhat.T @ y) + y.T @ y)/len(SLP_yhat)

            # Store argminimizer of MSE
            if tmpMSE < SLP_minMSE:
                SLP_minMSE=tmpMSE
                SLP_minMSEIndex=[0,i,j,k]
                SLP_minMSEBetas=SLP_betas
                

SLP_minMSEVariables=[variables[v] for v in SLP_minMSEIndex]

print("The sparse linear predictor results in {} with a MSE of {:.5f}.".format(SLP_minMSEVariables[1:],SLP_minMSE.item()))


'''
Solution to 2b: Sparse linear predictor betas
'''
print("{:<7} {:<20} {:<10}".format('Index','Variable','Coefficient'))
print("-"*40)

for i in range(4):
    print(" {:<7} {:<20} {:.4f}".format(SLP_minMSEIndex[i], SLP_minMSEVariables[i],SLP_minMSEBetas[i].item()))





# Solution to 2a: Test risk for the sparse linear predictor

SLP_yhat = testdata[:,SLP_minMSEIndex] @ SLP_minMSEBetas
SLP_MSE = (SLP_yhat.T @ SLP_yhat - 2*(SLP_yhat.T @ testlabels) + testlabels.T @ testlabels)/len(SLP_yhat)

print("The test risk for the sparse linear predictor is {:.4f}.".format(SLP_MSE.item()))





#---------Highest Correlatd Variables--------

# Ignore runtime message for finding correlation of constant variable
np.seterr(divide='ignore', invalid='ignore')

# Closest Correlation
closestCorr = dict()
corrMatrix = np.nan_to_num(np.corrcoef(testdata.T))


for i in range(1,numVariables):
    # Finds the indexes with the top 3 absolute correlations
    corrArgMax = np.argsort([abs(i) for i in corrMatrix[i]])[-3:][::-1]
    
    # Add each variable to a dictionary - Add a dictionary of the top two correlated variables and their correlation coeffs
    closestCorr[variables[i]]={variables[corrArgMax[1]]:corrMatrix[i][corrArgMax[1]],
                               variables[corrArgMax[2]]:corrMatrix[i][corrArgMax[2]]}


# Solution to 2c: highest variable correlations

print("{:<25} {:<8} {:<25} {:<10}".format('Variable','Rank','Most Correlated With','Correlation Coeff'))
print("-"*78)

for i in closestCorr.keys():
    rank=0
    for j in closestCorr[i].keys():
        rank+=1
        print(" {:<25} {:<8} {:<25} {:.5f}".format(i, rank, j, closestCorr[i][j]))
    print("-"*78)
    
print("\n  Note: the affine expansion variable is not included, because it has a\n\tcorrelation of 0 with all variables.")

