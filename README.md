#Answers (Draft)

###### Team member:
- Brian Allen (ba2542)
- Haozheng Ni (hn2318)
- Serena Zhang (mz2642)

Q1(Serena):

* (a) To find the vector xi that has the minimal Euclidean distance from x, it suffices to find xi that has the minimal squared Euclidean distance. Therefore, for each row tj in test data, we calculate its squared Euclidean distance to each row xi in training data and find xi\* that yields the smallest result. To achieve this, we leverage the given formula for the calculation:

 * for each tj in test data, we get the last term of the formula X\*X.T by obtaining the diagnal of the dot product of X and X.T
 * for the second term -2T.T\*X, we simply take the dot product of X and T.T
 * as the first term is the same for every tj in the test data, we can ignore this term in our calculation

In the end, for each tj, we obtain an array of the squared Euclidean distance between tj and each xi, and we'd like to pick i\* which corresponds to the smallest element in this array. 

As a final step, we predict yi\* for tj by using index i*.
* (b) See plot
* (c) It would be a horizontal line at test error rate = 0 as the nearest neighbor for data in the training set is itself, which results in training error rate of 0.

Q2(Brian):

Q3&Q4:latex file(Haozheng)
