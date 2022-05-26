import numpy as np
class LinearRegression:
    
    def __init__(self,lr=0.001,n_iters=1000):
        self.lr=lr
        self.n_iters=n_iters
        self.slope=None
        self.intercept=None
    
    def fit(self,X,y):
        n_samples,n_features =X.shape
        self.slope=np.zeros(n_features)
        self.intercept=0
        
        for _ in range(self.n_iters):
            y_pred=np.dot(X,self.slope)+self.intercept
            dw=(2/n_samples)*np.dot(X.T,(y_pred-y))
            
            db=(2/n_samples)*np.sum((y_pred-y))
            
            self.slope -=self.lr*dw
            self.intercept -=self.lr*db
            print(self.slope,self.intercept)

    def predict(self,X):
        pred=np.dot(X,self.slope)+self.intercept
        return pred
        