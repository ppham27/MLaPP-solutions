import numpy as np
import matplotlib.pyplot as plt

def plot_xy(x, y, ax=None):
    if ax == None:
        ax = plt.gca()    
    ax.scatter(x, y)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Training data")
    ax.grid(True)

def plot_abline(slope, intercept, xmin, xmax, ax=None):
    if ax == None:
        ax = plt.gca()
    ax.plot([xmin, xmax], [xmin*slope + intercept, xmax*slope + intercept],
            linewidth=3, color='red')

class SimpleOnlineLinearRegressor:
    ## keep track of sufficient statistics
    def __init__(self):
        self.N = 0
        self.x_sum = 0
        self.y_sum = 0
        self.x_squared_sum = 0
        self.y_squared_sum = 0
        self.xy_sum = 0
        self.w0 = 0
        self.w1 = 0
        self.sigma2 = 0
    def predict(self, X):
        return self.w0 + self.w1*X
    def fit(self, X, y):
        cov = np.cov(X,y,bias=True)
        self.N = len(y)
        self.w1 = cov[0,1]/cov[0,0]
        self.w0 = np.mean(y) - self.w1*np.mean(X)
        self.sigma2 = np.dot(y - self.w0 - self.w1*X, y - self.w0 - self.w1*X)/self.N    
    def partial_fit(self, x, y):
        self.N += 1
        self.x_sum += x
        self.y_sum += y
        self.x_squared_sum += x*x
        self.y_squared_sum += y*y
        self.xy_sum += x*y  
        if self.N > 1:
            self.w1 = (self.xy_sum - self.x_sum*self.y_sum/self.N)/(self.x_squared_sum - self.x_sum*self.x_sum/self.N)
            self.w0 = (self.y_sum - self.w1*self.x_sum)/self.N
            self.sigma2 = self.w0*self.w0 + (self.y_squared_sum - 2*self.w0*self.y_sum - 2*self.w1*self.xy_sum + 2*self.w0*self.w1*self.x_sum + self.w1*self.w1*self.x_squared_sum)/self.N
    def get_params(self):
        return {'intercept': self.w0, 'slope': self.w1, 'variance': self.sigma2}
