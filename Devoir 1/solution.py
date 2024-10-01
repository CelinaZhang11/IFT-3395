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

    def feature_means(self, iris) :
        data = iris[:, :-1] 

        means = np.mean(data, axis=0) 

        return means

    def empirical_covariance(self, iris) :
        data = iris[:, :-1]

        cov = np.cov(data, rowvar=False)

        return cov

    def feature_means_class_1(self, iris) :
        data_with_labels = iris
        
        # Select the rows with label 1
        label_1 = data_with_labels[data_with_labels[:, -1] == 1]
        
        # Exclude the label column and calculate the means for the first 4 columns
        means = np.mean(label_1[:, :-1], axis=0)

        return means

    def empirical_covariance_class_1(self, iris) : 
        data_with_labels = iris
    
        # Select the rows with label 1
        label_1 = data_with_labels[data_with_labels[:, -1] == 1]
    
        # Exclude the label column and calculate the covariance for the first 4 columns
        cov = np.cov(label_1[:, :-1], rowvar=False)
    
        return cov

def manhattan_distance(p, X) :
    return np.sum(np.abs(X - p), axis=1)  

class HardParzen :
    def __init__(self, h) :
        self.h = h

    def fit(self, train_inputs, train_labels) :
        self.label_list = np.unique(train_labels)
        self.train_inputs = train_inputs
        self.train_labels = train_labels 

    def predict(self, test_data) :
        # Initialization of the count matrix and the predicted classes array
        num_test = test_data.shape[0]
        counts = np.zeros((num_test, self.label_list))
        classes_pred = np.zeros(num_test)

        # For each test datapoint
        for i, p in enumerate(test_data) :
            # Calculate distances to each training set point using Manhattan distance
            distances = self.manhattan_distance(p, self.train_inputs)
            M = len(distances)

            # Go through the training set to find neighbors of the current point (p)
            ind_neighbors = []
            radius = self.h
            while len(ind_neighbors) == 0 :
                ind_neighbors = np.array([j for j in range(M) if distances[j] < radius])
                radius *= 2

            # If no neighbors are found, draw a random label
            if len(ind_neighbors) == 0 :
                label = self.draw_rand_label(p, self.label_list)
                classes_pred[i] = label
            else:
                # Get the labels of the neighbors
                cl_neighbors = list(self.train_labels[ind_neighbors] - 1)
                for j in range(len(cl_neighbors)) :
                    counts[i, cl_neighbors[j]] += 1

                # Assign most frequent label
                classes_pred[i] = np.argmax(counts[i, :]) + 1

        return classes_pred
        

class SoftRBFParzen:
    def __init__(self, sigma) :
        self.sigma  = sigma

    def fit(self, train_inputs, train_labels) :
        self.label_list = np.unique(train_labels)
        self.train_inputs = train_inputs
        self.train_labels = train_labels

    def rbf_kernel(self, distance) :
        return np.exp(-distance**2/(2*self.sigma**2))

    def predict(self, test_data) :
        # Initialization of the count matrix and predicted classes array
        num_test = test_data.shape[0]
        counts = np.zeros((num_test, len(self.label_list)))
        classes_pred = np.zeros(num_test) 

        # For each test datapoint
        for i, p in enumerate(test_data) :
            distances = manhattan_distance(p, self.train_inputs)

            # Calculate the weights for each training point
            weights = self.rbf_kernel(distances)

            # Accumulate the weights for each class label
            for j, label in enumerate(self.label_list) :
                class_indices = np.where(self.train_labels == label)[0]
                counts[i, j] = np.sum(weights[class_indices])

            # Assign the class with the highest weigth
            classes_pred[i] = self.label_list[np.argmax(counts[i, :])]

        return classes_pred    

def split_dataset(iris):
    # Get the indices of the rows
    indices = np.arange(iris.shape[0])

    # Remainder is 0, 1 or 2
    train = iris[indices % 5 <= 2]

    # Remainder is 3
    validation = iris[indices % 5 == 3]

    # Remainder is 4
    test = iris[indices % 5 == 4]

    return train, validation, test


class ErrorRate :
    def __init__(self, x_train, y_train, x_val, y_val) :
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val

    def hard_parzen(self, h) :
        hp = HardParzen(h)
        hp.fit(self.x_train, self.y_train)

        # Make predictions
        predictions = hp.predict(self.x_val)
        
        # Calculate error rate
        error_rate = 0
        for i, prediction in enumerate(predictions) :
            if prediction != self.y_val[i] :
                error_rate += 1
        
        return error_rate / len(predictions)

    def soft_parzen(self, sigma) :
        sp = SoftRBFParzen(sigma)
        sp.fit(self.x_train, self.y_train)

        # Make predictions
        predictions = sp.predict(self.x_val)
        
        # Calculate error rate
        error_rate = 0
        for i, prediction in enumerate(predictions) :
            if prediction != self.y_val[i] :
                error_rate += 1
        
        return error_rate / len(predictions)


def get_test_errors(iris):
    pass


def random_projections(X, A):
    pass


if __name__ == "__main__":

    # Test Q1
    Q1 = Q1()

    print(Q1.feature_means(iris))
    print(Q1.empirical_covariance(iris))
    print(Q1.feature_means_class_1(iris))
    print(Q1.empirical_covariance_class_1(iris))

    # Test split_dataset
    train, validation, test = split_dataset(iris)

    # Display the first few rows of each set
    print("Train set:\n", train[:5])
    print("Validation set:\n", validation[:5])
    print("Test set:\n", test[:5])