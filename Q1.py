'''
Team member:
Brian Allen (ba2542)
Haozheng Ni (hn2318)
Serena Zhang (mz2642)
'''
#---------Set up --------
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import random
import time

# load data
ocr = loadmat('ocr.mat')

#---------Calculate Nearest Neighbor--------
'''
# function:nearest neighbors classifier given different training data
# input: list; sampling size of training data for each iteration of prediction
# output: list; prediction outcome for test data
'''
def sampling(steps):
    test_errors=[]
    for n in steps:
        #sample n data points from training data
        sel=random.sample(range(60000),n)
        trainx=ocr["data"][sel].astype("float")
        trainy=ocr["labels"][sel].astype("float")
        testx=ocr["testdata"].astype("float")
        #predict nearest neighbors of test data using matrix caulculation
        #store the index of y data in var "yindex"
        result=np.array([])
        i=np.array([np.einsum('ij,ji->i', trainx,trainx.T)]).T
        k=np.matmul(trainx,testx.T)
        squared=i-2*k
        yindex=np.argmin(squared, axis=0)
        result=np.append(result,trainy[yindex])
        #calculate the test error rate
        #store it in list "test_errors"
        predicted=trainy[yindex]
        actual=ocr["testlabels"]
        test_errors.append((np.sum(predicted!=actual))/ocr["testlabels"].shape[0])
    return test_errors
'''
call the sampling function 10 times
store results in list "avg"
'''
final=[]
steps=[1000,2000,4000,8000]
start_time = time.time()
for i in range(10):
    temp=np.asarray(sampling(steps))
    final.append(temp)
avg=[sum(e)/len(e) for e in zip(*final)]
mat=np.vstack(final)
st=np.std(mat,axis=0)
print("The model ran for %s seconds" % (time.time() - start_time))
print("The means are: ", avg)
print("The standard deviations for each step are: ", st)

#---------Plot Test Error Rate--------
# plot the averaged test error rate with sample size increasing
plt.errorbar(steps,avg,yerr=st,fmt="-0")
plt.grid()
plt.xlabel("sample size")
plt.ylabel("test error rate")
plt.title("Test Error Rate vs Sample Size")
plt.savefig('Learning Curve Plot.png')
plt.show()

