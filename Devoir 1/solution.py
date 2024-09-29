import numpy as np
from collections import Counter
iris = np.genfromtxt("iris.txt")

######## DO NOT MODIFY THIS FUNCTION ########
def draw_rand_label(x, label_list):
    seed = abs(np.sum(x))
    while seed < 1:
        seed = 10 * seed
    seed = int(1000000 * seed)
    np.random.seed(seed)
    return np.random.choice(label_list)
#############################################


class Q1:

    def feature_means(self, iris):

        data = iris[:, :-1] 

        means = np.mean(data, axis=0) 

        return means

    def empirical_covariance(self, iris):
        
        data = iris[:, :-1]

        cov = np.cov(data, rowvar=False)

        return cov

    def feature_means_class_1(self, iris):

        data_with_labels = iris
        
        # Select the rows with label 1
        label_1 = data_with_labels[data_with_labels[:, -1] == 1]
        
        # Exclude the label column and calculate the means for the first 4 columns
        means = np.mean(label_1[:, :-1], axis=0)

        return means

    def empirical_covariance_class_1(self, iris):
            
        data_with_labels = iris
    
        # Select the rows with label 1
        label_1 = data_with_labels[data_with_labels[:, -1] == 1]
    
        # Exclude the label column and calculate the covariance for the first 4 columns
        cov = np.cov(label_1[:, :-1], rowvar=False)
    
        return cov


class HardParzen:
    def __init__(self, h):
        self.h = h

    def fit(self, train_inputs, train_labels):
        self.label_list = np.unique(train_labels)
        self.train_inputs = train_inputs
        self.train_labels = train_labels

    def manhattan_distance(self, p, X):
        return np.sum(np.abs(X - p), axis=1)   

    def predict(self, test_data):
        predictions = []

        for p in test_data :

            # Calculate the distances between the test point and all the training points
            distances = self.manhattan_distance(p, self.train_inputs)

            # Select the neighbors within the radius h
            neighbors = np.where(distances <= self.h)[0]

            if len(neighbors) == 0 :
                # If there are no neighbors, draw a random label
                label = draw_rand_label(p, self.label_list)
            else :
                # Get the labels of the neighbors
                neighbor_labels = self.train_labels[neighbors]
                label = Counter(neighbor_labels).most_common(1)[0][0]

            predictions.append(label)    
        
        return np.array(predictions)
        

class SoftRBFParzen:
    def __init__(self, sigma):
        self.sigma  = sigma

    def fit(self, train_inputs, train_labels):
        # self.label_list = np.unique(train_labels)
        pass

    def predict(self, test_data):
        pass


def split_dataset(iris):
    pass


class ErrorRate:
    def __init__(self, x_train, y_train, x_val, y_val):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val

    def hard_parzen(self, h):
        pass

    def soft_parzen(self, sigma):
        pass


def get_test_errors(iris):
    pass


def random_projections(X, A):
    pass


if __name__ == "__main__":
    Q1 = Q1()

    print(Q1.feature_means(iris))
    print(Q1.empirical_covariance(iris))
    print(Q1.feature_means_class_1(iris))
    print(Q1.empirical_covariance_class_1(iris))
    