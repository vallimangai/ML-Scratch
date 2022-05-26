import numpy as np

class LogisticRegression():
    
    def __init__(self,lr=0.001,n_iters=1000):
        self.lr=lr
        self.n_iters=n_iters
        self.slope=None
        self.intercept=None
      
    def fit(self,X,y):
        n_samples,n_features=X.shape
        self.slope=np.zeros(n_features)
        self.intercept=0
        
        #gradient descent
        for _ in range(self.n_iters):
            linearmodel=np.dot(X,self.slope)+self.intercept
            y_pred=self._sigmoid(linearmodel)
            
            dw=(2/n_samples)*(np.dot(X.T,(y_pred-y)))
            db=(2/n_samples)*(np.sum(y_pred-y))
            
            self.slope-=self.lr*dw
            self.intercept-=self.lr*db
        
    def predict(self,X):
        linearmodel=np.dot(X,self.slope)+self.intercept
        y_pred=self._sigmoid(linearmodel)
        y_predicted=[ 1 if i>0.5 else 0 for i in y_pred]
        return y_predicted
        
    def _sigmoid(self,x):
        return 1/(1+np.exp(-x))
        