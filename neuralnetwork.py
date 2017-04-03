"""Author: Nagabhushan S Baddi\nModel of a Neural Network used to learn the MNIST dataset"""

#import standard modules
from __future__ import division
import numpy as np
import os, sys
from math import *


class NeuralNetwork:
    """Neural Network Model"""
    def __init__(self, layers):
        self.layers = layers
        self.numLayers = len(layers)
        self.biases = [np.random.uniform(-4*sqrt(6/(layers[x-1]+layers[x])), 4*sqrt(6/(layers[x-1]+layers[x])), (layers[x], 1)) for x in range(1, self.numLayers)]
        self.weights = [np.random.uniform(-4*sqrt(6/(layers[x-1]+layers[x])), 4*sqrt(6/(layers[x-1]+layers[x])), (layers[x], layers[x-1])) for x in range(1, self.numLayers)]


    def feedforward(self, x):
        """feedforward the input through the neural network"""
        a = x
        for b, w in zip(self.biases, self.weights):
            a = self.sigmoid(np.dot(w, a)+b)

        return a


    def sigmoid(self, z):
        """compute the sigmoid function of z"""
        return 1.0/(1.0+np.exp(-z))


    def sigmoidPrime(self, z):
        """compute the derivative of sigmoid function"""
        return self.sigmoid(z)*(1-self.sigmoid(z))


    def SGM(self, trainingData, eta, epochs, miniBatchSize, regParam, testSet):
        """Obtain the minibatches and update the parameters for each minibatch"""
        n = len(trainingData)
        for epoch in range(epochs):
            np.random.shuffle(trainingData)
            miniBatches = [trainingData[k:k+miniBatchSize] for k in xrange(0, n, miniBatchSize)]
            for miniBatch in miniBatches:
                self.updataParameterForMiniBatch(eta, miniBatch, regParam)
            print("Epoch "+str(epoch)+": "+str(self.evaluate(testSet)))

    def updataParameterForMiniBatch(self, eta, miniBatch, regParam):
        """Update the weights and biases using stochastic gradient"""
        sumDeltaB = [np.zeros(b.shape) for b in self.biases]
        sumDeltaW = [np.zeros(w.shape) for w in self.weights]
        #print(miniBatch)
        for (x, y) in (miniBatch):
            deltaB, deltaW = self.propBack(x, y)
            sumDeltaB = [b1+b2 for b1, b2 in zip(deltaB, sumDeltaB)]
            sumDeltaW = [w1+w2 for w1, w2 in zip(deltaW, sumDeltaW)]
        self.biases = [b1-eta/len(miniBatch)*b2 for b1, b2 in zip(self.biases, sumDeltaB)]
        self.weights = [w1-eta*(w2/len(miniBatch)+regParam*w1) for w1, w2 in zip(self.weights, sumDeltaW)]


    def propBack(self, x, y):
        """x is the input vector and y is the output vector"""
        derB = [np.zeros(b.shape) for b in self.biases]
        derW = [np.zeros(w.shape) for w in self.weights]

        aList = [x]
        zList = []
        a = x
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, a)+b
            zList.append(z)
            a = self.sigmoid(z)
            aList.append(a)

        delta = (a-y)*self.sigmoidPrime(zList[-1])
        derB[-1] = delta
        derW[-1] = np.dot(delta, np.transpose(aList[-2]))

        for l in range(2, self.numLayers):
            delta = np.dot(np.transpose(self.weights[-l + 1]), delta) * self.sigmoidPrime(zList[-l])
            derB[-l] = delta
            derW[-l] = np.dot(delta, np.transpose(aList[-l - 1]))

        return (derB, derW)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]

        return sum(int(x == y) for (x, y) in test_results)/len(test_data)*100


import dataset

def Run(epochs, eta, miniBatchSize, regParam):
    net = NeuralNetwork([784, 30, 10])
    trainingSet, validationSet, testSet = dataset.load_data_wrapper()
    net.SGM(trainingSet, eta, epochs, miniBatchSize, regParam, testSet)
    print(net.evaluate(testSet))

if __name__ == "__main__":
    Run(30, 3, 10, 0)
