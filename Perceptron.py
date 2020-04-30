"""
  Basic Neural Network with No Hidden Layers (Perceptron)

  This perceptron takes in three inputs, all either 0 or 1.

"""

import numpy as np

class NeuralNetwork():
    def __init__(self):
        np.random.seed()
        self.synaptic_weights = 2 * np.random.random((3,1)) - 1

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, training_inputs, training_outputs, iterations):
        for iteration in range(10000):
            outputs = self.think(training_inputs)
            error = training_outputs - outputs

            if (error < 0.1).all():
                print("Training Iterations: ")
                print(iterations)
                break

            adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(outputs))
            self.synaptic_weights += adjustments

    def think(self, inputs):
        inputs = inputs.astype(float)
        outputs = self.sigmoid(np.dot(inputs, self.synaptic_weights))

        return outputs

if __name__ == "__main__":
    print("Inputs: A B C")
    print("Expected Output: A")
    x = "y"

    while(x == "y"):
        NeuralNet = NeuralNetwork()

        print("Synaptic Weights: ")
        print(NeuralNet.synaptic_weights)

        # Input = [A,B,C]  Output = A
        training_inputs = np.array([[0,0,1], [1,1,1], [1,0,1],[0,1,1]])
        training_outputs = np.array([[0,1,1,0]]).T

        NeuralNet.train(training_inputs, training_outputs, 100000)

        print("Synaptic Weights After Training: ")
        print(NeuralNet.synaptic_weights)

        A = str(input("Input 1: "))
        B = str(input("Input 2: "))
        C = str(input("Input 3: "))

        print("New Input Data = ", A, B, C)
        print("Output Data: ")
        print(NeuralNet.think(np.array([A,B,C])))

        x = input("Try Again? y/n: ")

    print("")
    print("Inputs: A B C")
    print("Expected Output: (A and B) or (B and C)")
    x = "y"
    while(x == "y"):
        NeuralNet = NeuralNetwork()

        print("Synaptic Weights: ")
        print(NeuralNet.synaptic_weights)


        # Input = [A,B,C] Output = (A and B) or (B and C)
        training_inputs = np.array([[0,0,1],[1,0,0],[1,1,0],[1,1,1],[0,1,1]])
        training_outputs = np.array([[0,0,1,1,1]]).T

        NeuralNet.train(training_inputs, training_outputs, 100000)

        print("Synaptic Weights After Training: ")
        print(NeuralNet.synaptic_weights)

        A = str(input("Input 1: "))
        B = str(input("Input 2: "))
        C = str(input("Input 3: "))

        print("New Input Data = ", A, B, C)
        print("Output Data: ")
        print(NeuralNet.think(np.array([A,B,C])))

        x = input("Try Again? y/n: ")


