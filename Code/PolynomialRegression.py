import numpy as np
from numpy.linalg import inv
import math


class PolynomialRegression:
    def __init__(self):
        self.d_final = 0
        self.w_final = 0
        self.loss = 0  
        self.avg_rmses = 0


    def determineWFinal(self, X, y):
        mean_X = np.mean(X)
        std_X = np.std(X)
        norm_X = self.normalize(X, mean_X, std_X)

        mean_y = np.mean(y)
        std_y = np.std(y)
        norm_y = self.normalize(y, mean_y, std_y)

        self.w_final = self.calculate_w(self.createMatrix(norm_X, self.d_final+1), norm_y)


    def fit(self, X, y, d_max, folds):
        rmse = np.zeros(d_max+1)

        for i in range(folds):
            len_test = math.floor(len(X)/folds)
            cross_val_train_X, cross_val_test_X = self.createCrossValSets(X.copy(), i, len_test)
            cross_val_train_y, cross_val_test_y = self.createCrossValSets(y.copy(), i, len_test)

            train_X_mean = np.mean(cross_val_train_X)
            train_X_std = np.std(cross_val_train_X)

            train_y_mean = np.mean(cross_val_train_y)
            train_y_std = np.std(cross_val_train_y) 

            normalized_train_X = self.normalize(cross_val_train_X, train_X_mean, train_X_std)
            normalized_train_y = self.normalize(cross_val_train_y, train_y_mean, train_y_std)
            normalized_test_X = self.normalize(cross_val_test_X, train_X_mean, train_X_std)

            for j in range(d_max+1):     
                w = self.calculate_w(self.createMatrix(normalized_train_X, j+1), normalized_train_y)
                rmse[j] = rmse[j] + self.RMSE(normalized_test_X, cross_val_test_y, w, j, train_y_mean, train_y_std)
     
        self.avg_rmses = rmse/folds
        self.d_final = np.argmin(rmse)


    def createCrossValSets(self, data, i, length): 
        return np.concatenate((data[:(length * i)], data[(length * (i + 1)):])),\
              data[(length * i):(length * (i + 1))]   


    def normalize(self, data, mean, std):
        return (data-mean)/std
    

    def denormalize(self, y, mean, std):
        return y * std + mean

    
    def createMatrix(self, data, d):
        matrix = list()
        for i in data:
            row = list()
            for j in range(d):
                row.append(np.power(i, j))
            matrix.append(row)
        return np.array(matrix, dtype=np.float32)


    def calculate_w(self, w, y):
        w_t = w.transpose()
        return np.matmul(np.matmul(inv(np.matmul(w_t,w)), w_t),y) 


    def RMSE(self, X, y, w, d, mean, std):
        sum = 0
        m = len(X)

        for i in range(m):
            sum = sum + np.square(y[i] - self.denormalize(self.predict(w, d, X[i]), mean, std))
        return np.sqrt(sum/m)
    

    def Loss(self, y_pred, y):
        m = len(y)
        sum = 0
        
        for i in range(m):
            sum = sum + np.square(y[i] - y_pred[i])
        self.loss = 0.5 * np.sqrt(sum/m)


    def predict_y(self, X, y,):
        mean_X = np.mean(X)
        std_X = np.std(X)

        X = self.normalize(X, mean_X, std_X)

        mean_y = np.mean(y)
        std_y = np.std(y)

        predictions = np.zeros(len(y))

        for i in range(len(y)):
            predictions[i] = self.denormalize(self.predict(self.w_final, self.d_final, X[i]), mean_y, std_y)
        return predictions


    def predict(self, w, d, x):
        sum = 0

        for j in range(d):
            sum = sum + w[j] * np.power(x, j)
        return sum