import numpy as np 
import data_classification_utils
from util import raiseNotDefined
import random

class Perceptron(object):
    def __init__(self, categories, numFeatures):
        """categories: list of strings 
           numFeatures: int"""
        self.categories = categories
        self.numFeatures = numFeatures
        self.weightsForCategories = {}
        for label in categories:
            self.weightsForCategories[label] = np.zeros(numFeatures)
        """Each category is a class"""


    def classify(self, sample):
        """sample: np.array of shape (1, numFeatures)
           returns: category with maximum score, must be from self.categories"""

        """YOUR CODE HERE"""
        maxScore = float("-inf")
        maxLabel = ""
        for label in self.categories:
            currScore = sample.dot(self.weightsForCategories[label])
            if (currScore) > maxScore:
                maxLabel = label
                maxScore = currScore
        return maxLabel

    def train(self, samples, labels):
        """samples: np.array of shape (numFeatures, numSamples)
           labels: list of numSamples strings, all of which must exist in self.categories 
           performs the weight updating process for perceptrons by iterating over each sample once."""

        """YOUR CODE HERE"""
        for i, sample in enumerate(samples):
            #classify, if classification is wrong then subtract
            classifiedLabel = self.classify(sample)
            if classifiedLabel != labels[i]:
                self.weightsForCategories[labels[i]] = self.weightsForCategories[labels[i]] + sample
                self.weightsForCategories[classifiedLabel] = self.weightsForCategories[classifiedLabel] - sample

