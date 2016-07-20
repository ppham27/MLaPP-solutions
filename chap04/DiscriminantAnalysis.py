import numpy as np
from scipy import stats

class QDA:
    def fit(self, X, y):
        assert(len(X) == len(y))
        self.n = len(y) # number of observations
        self.p = X.shape[1] # number of features
        self.classes = np.unique(y)
        self.C = len(self.classes)        
        self.theta = np.empty(self.C, dtype=np.float64)
        self.covariances = np.empty((self.C, self.p, self.p), dtype=np.float64)
        self.mu = np.empty((self.C, self.p), dtype=np.float64)
        for i in range(self.C):
            sub_X = X[y == self.classes[i]]            
            self.theta[i] = len(sub_X)/len(y)
            self.covariances[i] = np.cov(sub_X, rowvar=False, bias=False)
            self.mu[i] = np.mean(sub_X, axis=0)
        return self.mu
    def predict(self, X):
        if X.ndim == 1:
            X = X.reshape((1, len(X)))
        predictions = np.empty(len(X), dtype=self.classes.dtype)
        for i in range(len(X)):
            class_numerator = np.empty(self.C, dtype=np.float64)
            for c in range(self.C):
                class_numerator[c] = self.theta[c]*stats.multivariate_normal.pdf(x=X[i], 
                                                                                 mean=self.mu[c], 
                                                                                 cov=self.covariances[c])
            predictions[i] = self.classes[np.argmax(class_numerator)]
        return predictions
    def score(self, X, y):
        predictions = self.predict(X)
        return sum(predictions == y)/len(y)

class LDA:
    def fit(self, X, y):
        assert(len(X) == len(y))
        self.n = len(y) # number of observations
        self.p = X.shape[1] # number of features
        self.classes = np.unique(y)
        self.C = len(self.classes)        
        self.theta = np.empty(self.C, dtype=np.float64)
        self.covariance = np.zeros((self.p, self.p), dtype=np.float64)
        self.mu = np.empty((self.C, self.p), dtype=np.float64)
        for i in range(self.C):
            sub_X = X[y == self.classes[i]]                        
            self.theta[i] = len(sub_X)/len(y)            
            self.mu[i] = np.mean(sub_X, axis=0)
            self.covariance += np.cov(sub_X, rowvar=False, ddof=len(sub_X) - 1)
        self.covariance /= self.n
        return self.mu
    def predict(self, X):
        if X.ndim == 1:
            X = X.reshape((1, len(X)))
        predictions = np.empty(len(X), dtype=self.classes.dtype)
        for i in range(len(X)):
            class_numerator = np.empty(self.C, dtype=np.float64)
            for c in range(self.C):
                class_numerator[c] = self.theta[c]*stats.multivariate_normal.pdf(x=X[i], 
                                                                                 mean=self.mu[c], 
                                                                                 cov=self.covariance)
            predictions[i] = self.classes[np.argmax(class_numerator)]
        return predictions
    def score(self, X, y):
        predictions = self.predict(X)
        return sum(predictions == y)/len(y)
