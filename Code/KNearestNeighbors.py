import numpy as np
import math

class Knn:
    def __init__(self):
        self.k_final = 0
        self.loss = 0
        self.avg_rmses = 0

    def normalize(self, data, mean, std):
        return (data-mean)/std
    

    def denormalize(self, data, mean, std):
        return data * std + mean


    def findMaxIndex(self, distances):
        max = -9999999
        index = 0

        for i in range(len(distances)):
            if distances[i] > max:
                max = distances[i]
                index =  i 
        return index


    def createSets(self, data, length, i):
        return np.concatenate((data[:(length * i)], data[(length * (i + 1)):])),\
              data[(length * i):(length * (i + 1))]


    def train(self, train_X, train_y, test_X, k):
        predictions = np.zeros(len(train_X))
        for m in range(len(test_X)):
            closest = np.zeros(k)
            distances = np.zeros(k)
            index = 0
            index_largest = -1
            for l in range(len(train_X)):
                delta_x = np.square(test_X[m]-train_X[l])
                distance = np.sqrt(delta_x)
                
                if len(closest) < k:
                    distances[index] = distance
                    closest[index] = train_y[l]
                    if index_largest == -1:
                        index_largest = 0
                    else:
                        if distance < distances[index_largest]:
                            index_largest = index
                    index+=1
                else:
                    # max_index = self.findMaxIndex(distances)
                    distances[index_largest] = distance
                    closest[index_largest] = train_y[l]
            
            sum = np.sum(closest)
            predictions[m] = sum / len(closest)
            
        return np.array(predictions, dtype=np.float32)

    
    def fit(self, X, y, folds, k_min, k_max):
        n = k_max-k_min+1
        rmse = np.zeros(n)
        
        for i in range(folds):
            length_test_set = math.floor(len(X)/folds)
            
            cross_val_train_X, cross_val_test_X = self.createSets(X.copy(), length_test_set, i)
            cross_val_train_y, cross_val_test_y = self.createSets(y.copy(), length_test_set, i)
   
            train_X_mean = np.mean(cross_val_train_X)
            train_X_std = np.std(cross_val_train_X)

            train_y_mean = np.mean(cross_val_train_y)
            train_y_std = np.std(cross_val_train_y)

            norm_train_X = self.normalize(cross_val_train_X, train_X_mean, train_X_std)
            norm_train_y = self.normalize(cross_val_train_y, train_y_mean, train_y_std)

            norm_test_X = self.normalize(cross_val_test_X, train_X_mean, train_X_std)

            for j in range(k_min,k_max+1):
                predictions = self.train(norm_train_X, norm_train_y, norm_test_X, j+1)
                
                sum = 0
                for l in range(len(cross_val_test_y)):
                    sum = sum + np.square(cross_val_test_y[l] - self.denormalize(predictions[l],train_y_mean, train_y_std))
                rmse[j-k_min] = rmse[j-k_min] + np.sqrt(sum)

        self.avg_rmses = rmse / folds
        self.k_final = np.argmin(rmse)


    def Loss(self, train_X, train_y, test_X, test_y):
        train_X_mean = np.mean(train_X)
        train_X_std = np.std(train_X)

        train_y_mean = np.mean(train_y)
        train_y_std = np.std(train_y)

        norm_train_X = self.normalize(train_X, train_X_mean, train_X_std)
        norm_train_y = self.normalize(train_y, train_y_mean, train_y_std)

        norm_test_X = self.normalize(test_X, train_X_mean, train_X_std)

        predictions = self.train(norm_train_X, norm_train_y, norm_test_X, self.k_final)
                
        sum = 0
        for l in range(len(test_y)):
            sum = sum + np.square(test_y[l] - self.denormalize(predictions[l], train_y_mean, train_y_std))
        self.loss = 0.5 * np.sqrt(sum)

    