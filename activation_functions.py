# Video 5 Activation Functions:
# Activation functions trigger neurons to activate and calculating the end results
# That's also the reason you need 2 or more hidden layers because you cant have a neuron activate another when there is only one layer
import numpy as np

# A very simple rectified linear activation function (Video 5: 0:00 - ~22:00)
class ActivationReLU:
    def forward(self, inputs):  # Taking the input of the neuron "behind it"
        self.inputs = inputs  # To remember the input values
        self.output = np.maximum(0,
                                 inputs)  # Basically if the input is greater zero the output is 1 and if the number is 0 or less the output is 0

    def backward(self, dvalues):  # S.210
        # Since we need to modify the original variable, let's make a copy of the values first
        self.dinputs = dvalues.copy()
        # Zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0

    # Calculate predictions for outputs
    def predictions(self, outputs):
        return outputs

# Sigmoid activation
class ActivationSigmoid:
    # Forward pass
    def forward(self, inputs):
        # Save input and calculate/save output
        # of the sigmoid function
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    # Backward pass
    def backward(self, dvalues):
        # Derivative - calculates from output of the sigmoid function
        self.dinputs = dvalues * (1 - self.output) * self.output

    # Calculate predictions for outputs
    def predictions(self, outputs):
        return (outputs > 0.5) * 1

class ActivationLinear:
    # Forward pass
    def forward(self, inputs):
        # Just remember values
        self.inputs = inputs
        self.output = inputs

    # Backward pass
    def backward(self, dvalues):
        # derivative is 1, 1 * dvalues = dvalues - the chain rule
        self.dinputs = dvalues.copy()

    # Calculate predictions for outputs
    def predictions(self, outputs):
        return outputs

class ActivationSLCC:  # S.230
    def backward(self, dvalues, y_true):
        samples = len(dvalues)  # Number of samples
        if len(y_true.shape) == 2:  # If labels are one-hot encoded, turn them into discrete values
            y_true = np.argmax(y_true, axis=1)

        self.dinputs = dvalues.copy()  # Copy so we can safely modify
        self.dinputs[range(samples), y_true] -= 1  # Calculate gradient
        self.dinputs = self.dinputs / samples  # Normalize gradient

class ActivationSoftmax:
    def forward(self, inputs):  # Takes inputs of neurons and calculates gradients (probabilities)
        self.inputs = inputs  # To remember the input values
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, dvalues):  # S.216 - 225
        # Create an array that will later be filled out with the gradients to pass backwards
        self.dinputs = np.empty_like(dvalues)

        # Enumerate over outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)

            # Calculate Jacobian matrix of the output
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)

            # Calculate gradient and add it to the array of gradients
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

    # Calculate predictions for outputs
    def predictions(self, outputs):
        return np.argmax(outputs, axis=1)
