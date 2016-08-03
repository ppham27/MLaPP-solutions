import numpy as np
from scipy import io
import pandas as pd

def read_spam_data():
    raw_data = io.loadmat('spamData.mat')
    with open('spambase.names') as f:
        current_line = f.readline().strip()
        ## get response line
        while current_line == '' or current_line[0] == '|':
            current_line = f.readline().strip()
        split_response = current_line.split('|')
        split_response = list(map(lambda x : x.split(), split_response))
        for i in range(len(split_response)):
            split_response[i] = list(map(lambda x : x.strip(',.'), split_response[i]))
        names = dict()
        names['response'] = dict()
        for i in range(len(split_response[0])):
            names['response'][split_response[0][i]] = split_response[1][i]
        names['X_columns'] = list()
        for line in f:
            l = line.strip()
            if l != '' and l[0] != '|':
                names['X_columns'].append(l.split(':')[0])
    train_data = pd.DataFrame(raw_data['Xtrain'], columns=names['X_columns'])
    train_data.insert(0, column=names['response']['1'], value=raw_data['ytrain'].astype(np.int32))
    test_data = pd.DataFrame(raw_data['Xtest'], columns=names['X_columns'])
    test_data.insert(0, column=names['response']['1'], value=raw_data['ytest'].astype(np.int32))
    return train_data, test_data

def transform_log(x):
    return np.log(x + 0.1)

def transform_binary(x):
    return (x > 0).astype(np.int32)

class LogisticRegression:
    """
    Implements a subset of LogisticRegression from sklearn.
    My regularization parameter is not the reciprocal, so the implementation
    """
    def __init__(self, regularization = 1, max_iter=100):
        self.regularization = regularization
        self.max_iter = max_iter
    def fit(self, X, y):                
        X_tilde = np.column_stack((np.ones(len(X)), X))
        I = np.eye(X_tilde.shape[1])
        theta = np.zeros(X_tilde.shape[1])
        mu = lambda theta : 1/(1 + np.exp(-np.dot(X_tilde, theta)))
        # newton's method
        def g(theta):
            grad = np.dot(X_tilde.transpose(), mu(theta) - y) + self.regularization*theta
            grad[0] -= self.regularization*theta[0]
            return grad
        def H(theta):
            hessian = np.dot(X_tilde.transpose()*(mu(theta)*(1-mu(theta))), X_tilde) + I*self.regularization
            hessian[0,0] -= self.regularization        
            return hessian
        previous_theta = theta
        theta = theta + np.linalg.solve(H(theta), -g(theta))
        i = 0
        while np.sqrt(np.dot(theta - previous_theta, theta - previous_theta)) > 1e-6 and i < self.max_iter:
            previous_theta = theta
            theta = theta + np.linalg.solve(H(theta), -g(theta))
            i += 1
        self.intercept_ = theta[0]
        self.coef_ = theta[1:]
        return theta    
    def predict_proba(self, X):
        mu = 1/(1 + np.exp(-(self.intercept_ + np.dot(X, self.coef_))))
        return np.column_stack((1-mu,mu))
    def predict(self, X):
        return np.round(self.predict_proba(X)[:,1])
    def score(self, X, y):
        return sum(self.predict(X) == y)/len(y)

