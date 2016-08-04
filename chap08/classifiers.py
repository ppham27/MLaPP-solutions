import numpy as np
from scipy import io
from scipy import stats
from scipy.misc import logsumexp
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

class BernoulliNB:
    """
    Implements BernoulliNB from sklearn.
    There are some significant differences in implementation here.
    Mainly, my version let's once specify a Dirichlet prior on the class
    probabilities.
    """

    def __init__(self, alpha = 1, gamma = None):
        self.alpha = alpha
        self.gamma = gamma
        self.fitted = False

    def fit(self, X, y):
        if self.fitted is False: # first fit, do some initialization
            self.K = len(np.unique(y)) # assume y takes values 0,1,...,K - 1
            self.N = 0
            self.p = X.shape[1]
            self.class_counts = np.zeros(self.K)
            self.feature_counts = np.zeros((self.K, self.p))
            if self.gamma is None:
                self.gamma = np.zeros(self.K)
            self.fitted = True
        self.N += len(X)
        for i in range(len(X)):
            k = y[i]
            self.class_counts[k] += 1
            for j in range(self.p):
                if X[i][j] == 1:
                    self.feature_counts[k][j] += 1
    
    def predict_log_proba(self, X):
        log_probs = np.empty((len(X), self.K))
        for k in range(self.K):
            log_probs[:, k] = np.log(self.class_counts[k] + self.gamma[k]) - np.log(self.N + np.sum(self.gamma))
            for j in range(self.p):
                log_probs[X[:, j] == 1, k] += np.log(self.feature_counts[k, j] + self.alpha)
                log_probs[X[:, j] == 0, k] += np.log(self.class_counts[k] - self.feature_counts[k, j] + self.alpha)
                log_probs[:, k] -= np.log(self.class_counts[k] + 2*self.alpha)
        for i in range(len(X)):
            log_probs[i, :] -= logsumexp(log_probs[i, :])
        return log_probs

    def predict_proba(self, X):
        return np.exp(self.predict_log_proba(X))

    def predict(self, X):
        return np.apply_along_axis(np.argmax, axis = 1, arr=self.predict_log_proba(X))

    def score(self, X, y):
        return np.sum(self.predict(X) == y)/len(y)

class GaussianNB:
    def __init__(self, gamma = None):
        self.gamma = gamma
        self.fitted = False
        
    def fit(self, X, y):
        if self.fitted is False:
            self.K = len(np.unique(y))
            self.N = 0
            self.p = X.shape[1]
            self.class_counts = np.zeros(self.K)
            self.feature_sums = np.zeros((self.p, self.K))
            self.feature_squared_sums = np.zeros((self.p, self.K))
            if self.gamma is None:
                self.gamma = np.zeros(self.K)
            self.fitted = True
        self.N += len(y)
        for i in range(len(y)):
            k = y[i]
            self.class_counts[k] += 1
            for j in range(self.p):
                self.feature_sums[j, k] += X[i, j]
                self.feature_squared_sums[j, k] += X[i, j]*X[i, j]
    
    def predict_log_proba(self, X):
        log_probs = np.empty((len(X), self.K))
        for k in range(self.K):
            log_probs[:, k] = np.log(self.class_counts[k] + self.gamma[k]) - np.log(self.N + np.sum(self.gamma))
            for i in range(len(X)):
                for j in range(self.p):
                    mu_k = self.feature_sums[j, k]/self.class_counts[k]
                    log_probs[i, k] += stats.norm.logpdf(X[i, j], 
                                                         loc=mu_k,
                                                         scale=np.sqrt(self.feature_squared_sums[j, k]/self.class_counts[k] - mu_k*mu_k))
        for i in range(len(X)):
            log_probs[i, :] -= logsumexp(log_probs[i, :])
        return log_probs
    
    def predict_proba(self, X):
        return np.exp(self.predict_log_proba(X))

    def predict(self, X):
        return np.apply_along_axis(np.argmax, axis = 1, arr=self.predict_log_proba(X))

    def score(self, X, y):
        return np.sum(self.predict(X) == y)/len(y)
    
