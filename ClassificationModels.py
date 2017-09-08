#! /usr/bin/env python

########################################
# Matt Baer, Matt Parker
########################################

import csv
from sklearn.svm import SVC
from numpy import array, array_split, where, isnan, isinf, delete
from random import shuffle, choice
from numpy.linalg import norm
import operator
import warnings
warnings.filterwarnings("ignore")

class Classifier(object):
    """Base class for KNN, naive Bayes and SVM classifiers.

    Derived classes are expected to implement the train() and predict() methods.

    The test() and cross_validate() methods can be identical across all
    derived classes and should therefore be implemented here.
    """
    def __init__(self, data_file):
        """
        data_file is the name of a file containing comma-separated values
        with one observation per line. All but the last value one each line
        is treated as x, the input to the learned function. The last value
        is treated as y, the output of the learned function.
        The member variables x_data and y_data are lists with corresponding
        indices.
        """
        with open(data_file) as f:
            data = [map(float, l) for l in csv.reader(f)]
        self.x_data = [array(d[:-1]) for d in data]
        self.y_data = [d[-1] for d in data]

    def test(self, x_test, y_test):
        """Reports the fraction of the test set that is correctly classified.

        The predict() method is run on each element of x_test, and the result
        is compared to the corresponding element of y_test. The fraction that
        are classified correctly is tracked and returned as a float.
        """
        correct = 0
        elements = 0
        element_index = 0
        for element_index in range(len(x_test)):
            if self.predict(x_test[element_index]) == y_test[element_index]:
                correct += 1
            elements += 1

        print float(correct)/elements
        return float(correct)/elements


    def cross_validation(self, folds=3, params=[]):
        """Performs k-fold cross validation to select parameters.

        The x_data and y_data arrays are shuffled (preserving their
        correspondence) then divided into %folds subsets. For each parameter
        value, the classifier is trained and tested %folds times, each time,
        the training set is all but one of the subsets, and the test set is
        the one remaining subset. Error rate is averaged across these tests
        to give a score for the parameter value. The parameter value with the
        lowest average error is returned.
        """

        #If we zip the data arrays and shuffle the zip...it'd be sick
        """zipped_array = zip(self.x_data, self.y_data) #transforms x_data, y_data into pairs (x_data[i], y_data[i])
        print zipped_array
        shuffled_array = array((shuffle(zipped_array))) #shuffles the array of (x, y) tuples, then casts it to a numpy array
        print shuffled_array"""
        shuffled_indices = range(len(self.x_data))
        shuffle(shuffled_indices)
        x_data_shuf = array([self.x_data[i] for i in shuffled_indices])
        y_data_shuf =  array([self.y_data[i] for i in shuffled_indices])
        x_fold_sets = array_split(x_data_shuf, folds) #list of arrays of (x_data[i], y_data[i]) tuples
        y_fold_sets = array_split(y_data_shuf, folds)
        param_error = {}
        for param in params:
            for fold in range(1, folds):
                self.train(x_fold_sets[fold], y_fold_sets[fold], param)
            error = self.test(x_fold_sets[0], y_fold_sets[0])
            param_error[param] = error
        vals = param_error.values()
        keys = param_error.keys()
        return keys[vals.index(max(vals))]



class KNearestNeighbors(Classifier):
    def train(self, x_train, y_train, k=3):
        """Stores the training set after normalizing each dimension.

        k: number of neighbors considered when classifying a new point.

        Very little training is required here. The input data is stored in
        a list, but first each dimension is normalized to have values in the
        range 0-1. The normalization factors need to be stored so that they
        can be applied to the test set later.
        """
        self.x_train = x_train
        self.labels = y_train
        self.k = k
        self.normConstant = []
        # print self.x_train
        for featureIndex in range(len(x_train[0])):
            featureVector = self.x_train[:,featureIndex]
            if min(featureVector) == max(featureVector):
                minimum = 0
                maximum = 1
            else:
                minimum = min(featureVector)
                featureVector -= min(featureVector)
                maximum = max(featureVector)
            self.x_train[:,featureIndex] -= minimum
            self.x_train[:,featureIndex] /= maximum
            self.normConstant.append((minimum, maximum))  
        # print self.x_train  
        

    def predict(self, test_point):
        """Returns the plurality class over the k closest training points.

        Nearest neighbors are chosen by Euclidean distance (after normalizing
        the test point).
        Ties are broken by repeatedly removing the most-distant element(s)
        until a strict plurality is achieved.
        """
        #First, normalize test_point
        self.normalize(test_point)

        #Get neighbors
        neighbors = self.kNeighbors(test_point) #in form of array((distance,label))
        # print neighbors

        uniqueMostCommonLabel = False
        while uniqueMostCommonLabel == False:
            count = {}
            for x in range(len(neighbors)):
                response = neighbors[x][-1]
                if response in count:
                    count[response] += 1
                else:
                    count[response] = 1
            sortedCount = sorted(count.iteritems(), key=operator.itemgetter(1), reverse=True)
            # if len(sortedCount) > 1:
            #     print sortedCount[0][0], sortedCount[1][0]
            # print len(sortedCount)
            # print neighbors
            # print sortedCount

            if len(sortedCount) > 1:
                if sortedCount[0][1] == sortedCount[1][1]:
                    neighbors.pop()
                else:
                    uniqueMostCommonLabel = True
            else:
                uniqueMostCommonLabel = True

            
        # print sortedCount[0][0]
        return sortedCount[0][0]
       

    def normalize(self, test_point):
        for i in range(len(self.normConstant)):
            minimum, maximum = self.normConstant[i]
            test_point[i] -= minimum
            test_point[i] /= maximum

    def kNeighbors(self, test_point):
        """
        Assuming test_point is normalized
        """

        distances = []
        
        for x in range(len(self.x_train)):
            dist = norm(test_point - self.x_train[x])
            distances.append(dist)
        
        # print zip(distances,self.labels)

        neighbors = sorted(zip(distances,self.labels))
        
        #If there's a tie for kth nearest and (k+1)th nearest, add it
        done = False
        m = self.k
        while done == False:
            kneighbors = neighbors[:m]
            nextkneighbors = neighbors[:(m+1)]
            # print nextkneighbors[m-1][0], nextkneighbors[m][0]
            if m < len(kneighbors):
                if nextkneighbors[m-1][0] == nextkneighbors[m][0]:
                    m + 1
                else:
                    done = True
            else:
                done = True

        return kneighbors
        


class NaiveBayes(Classifier):
    def __init__(self, data_file):
        Classifier.__init__(self, data_file)
        """self.instances = 0
        self.combinations = {}
        self.label_instances = {}
        self.feature_value_instances = {}
        self.label_probabilities = {}"""

    def train(self, x_train, y_train, equiv_samples=10):

        """Computes the probability estimates for naive Bayes.

        Classifying an arbitrary test point requires an estimate of the
        following probabilities:
        - for each label l: P(l)
        - for each input dimension x_i and each value for that dimension:
            P(x_i = v | l) for each label l.
        These estimates are computed by combining the empirical frequency in
        the data set with a uniform prior.

        equiv_samples: determines how much weight to give to the uniform prior.

        The dimension of the input, the set of values for each input dimension,
        and the set of labels all need to be determined from the data set."""
        self.classification_probs = {}
        self.label_probs = self.get_output_probs(y_train, equiv_samples)
        for label in self.label_probs.keys():
            self.classification_probs[label] = {}
            for feature in range(len(x_train[0])):
                self.classification_probs[label][feature] = self.get_feature_probs_for_label(feature, label, x_train, y_train, equiv_samples)
        #print self.classification_probs

    def get_feature_probs_for_label(self, feature, label, x_train, y_train, samples):
        """returns a dict containing the adjusted probability of each value in the dataset for a given label"""
        self.values_for_feature = {}
        self.feature_value_probabilities = {}
        for data in range(len(x_train)):
            if y_train[data] == label:
                if x_train[data][feature] not in self.values_for_feature:
                    self.values_for_feature[x_train[data][feature]] = 1
                else:
                    self.values_for_feature[x_train[data][feature]] += 1
        for value in self.values_for_feature:
            self.feature_value_probabilities[value] = (self.values_for_feature[value] + (float(1)/len(self.values_for_feature.keys())) * samples)/float(sum(self.values_for_feature.values()) + samples)
        return self.feature_value_probabilities

    def get_output_probs(self, outputData, samples):
        self.output_freqs = {}
        self.output_probs = {}
        for data in outputData:
            if data in self.output_freqs:
                self.output_freqs[data] += 1
            else:
                self.output_freqs[data] = 1
        for label in self.output_freqs:
            self.output_probs[label] = (self.output_freqs[label] + (float(1)/len(self.output_freqs.keys())) * samples)/float(sum(self.output_freqs.values()) + samples)
        return self.output_probs

    def predict(self, test_point):
        """Returns the most probable label for test_point.

        Uses the stored probability of each label and conditional probabilities
        of test_point's input values from self.train()."""
        class_probs = {}
        for label in self.label_probs:
            class_probs[label] = self.label_probs[label] * self.computeFeatureProb(test_point, label)
        vals = class_probs.values()
        keys = class_probs.keys()
        return keys[vals.index(max(vals))]

    def computeFeatureProb(self, test_point, label):
        probability = 1
        for feature in self.classification_probs[label]:
            if test_point[feature] not in self.classification_probs[label][feature]:
                return 0
            else:
                probability *= self.classification_probs[label][feature][test_point[feature]]
        return probability






class SupportVectorMachine(Classifier):
    """Wrapper for the sklearn.svm.SVC classifier."""

    def train(self, x_train, y_train, kernel="linear"):
        """
        kernel: one of 'linear', 'poly', 'rbf', or 'sigmoid'.
        """
        self.svc_model = SVC(kernel=kernel)
        self.svc_model.fit(x_train, y_train)

    def predict(self, test_point):
        # SVC.predict takes one or many test points and always returns an array
        return self.svc_model.predict(test_point)[0]


def main():
    print "kNN on house_votes"
    knn = KNearestNeighbors("house_votes.data")
    print knn.cross_validation(params = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    print "kNN on spambase_boolean"
    knn = KNearestNeighbors("spambase_boolean.data")
    print knn.cross_validation(params = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    print "kNN on spambase"  
    knn = KNearestNeighbors("spambase.data")
    print knn.cross_validation(params = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    print "kNN on optdigits"
    knn = KNearestNeighbors("optdigits.data")
    print knn.cross_validation(params = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    

    
    print "NaiveBayes on house_votes"
    nb = NaiveBayes("house_votes.data")
    print nb.cross_validation(folds = 5, params = [1, 10, 100])

    print "NaiveBayes on spambase_boolean"
    nb = NaiveBayes("spambase_boolean.data")
    print nb.cross_validation(folds = 5, params = [1, 10, 100])

    print "NaiveBayes on spambase"
    nb = NaiveBayes("spambase.data")
    print nb.cross_validation(folds = 5, params = [1, 10, 100])

    print "NaiveBayes on optdigits"
    nb = NaiveBayes("optdigits.data")
    print nb.cross_validation(folds = 5, params = [1, 10, 100])


    print "SVM on house_votes"
    svm = SupportVectorMachine("house_votes.data")
    print svm.cross_validation(folds = 5, params = ["linear", "poly", "rbf", "sigmoid"])

    print "SVM on spambase_boolean"
    svm = SupportVectorMachine("spambase_boolean.data")
    print svm.cross_validation(folds = 5, params = ["linear", "poly", "rbf", "sigmoid"])

    print "SVM on spambase"
    svm = SupportVectorMachine("spambase.debug")
    print svm.cross_validation(folds = 2, params = ["linear", "poly", "rbf", "sigmoid"])
    

    print "SVM on optdigits"
    svm = SupportVectorMachine("optdigits.data")
    print svm.cross_validation(folds = 5, params = ["linear", "poly", "rbf", "sigmoid"])

    



if __name__ == "__main__":
    main()
