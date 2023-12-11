import numpy as np
from scipy import signal

# Class for creation of Input Layer, so we can loop through it in the model class
class LayerInput:
    def forward(self, inputs, training):
        self.output = inputs

class LayerHidden:
    def __init__(self, input_size, num_neurons, *, weight_regularizer_l1=0, weight_regularizer_l2=0, bias_regularizer_l1=0, bias_regularizer_l2=0):
        self.weights = 0.01 * np.random.randn(input_size, num_neurons)
        self.biases = np.zeros((1, num_neurons))

        # Set regularization strength
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

    def forward(self, inputs):
        self.inputs = inputs  # To remember the input values
        self.output = np.dot(inputs, self.weights) + self.biases  # Calculate the Number that will be passed into the next neurons

    def backward(self, dvalues):  # S.209
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # Gradients on regularization
        # L1 on weights
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1

        # L2 on weights
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights

        # L1 on biases
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1

        # L2 on biases
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases

        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)

    # Retrieve layer parameters
    def get_parameters(self):
        return self.weights, self.biases

    # Set weights and biases in a layer instance
    def set_parameters(self, weights, biases):
        self.weights = weights
        self.biases = biases


class LayerDropout:
    # Init
    def __init__(self, rate):
        # Store rate, we invert it as for example for dropout of 0.1 we need success rate of 0.9
        self.rate = 1 - rate

    # Forward pass
    def forward(self, inputs):
        # Save input values
        self.inputs = inputs
        # Generate and save scaled mask
        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        # Apply mask to output values
        self.output = inputs * self.binary_mask

    # Backward pass
    def backward(self, dvalues):
        # Gradient on values
        self.dinputs = dvalues * self.binary_mask

# Explanation: https://www.youtube.com/watch?v=Lakz2MoHy6o 1:40 - 27:18
class LayerConvolutional2D:
    def __init__(self, input_shape, kernel_shape):
        input_height, input_width, input_depth = input_shape   # depth = channels
        kernel_rows, kernel_col, num_kernels = kernel_shape

        self.num_kernels = num_kernels
        self.kernel_rows = kernel_rows
        self.kernel_col = kernel_col

        self.input_shape = input_shape
        self.input_depth = input_depth
        self.input_rows = input_height
        self.input_col = input_width

        self.output_shape = (num_kernels, self.input_rows - self.kernel_rows + 1, self.input_col - self.kernel_col + 1, self.input_depth)

        self.kernels_shape = (self.kernel_rows, self.kernel_col, self.num_kernels)

        self.kernels = np.random.randn(*self.kernels_shape)   # building the kernels randomly
        self.biases = np.random.randn(*self.output_shape)

    def forward(self, input):
        self.input = input

        # Initialize an empty result matrix
        self.output = np.zeros(self.output_shape)
        # self.output = self.output + np.copy(self.biases)
        self.window_batch = np.stack((self.input, self.kernels), axis=0)

        # turn kernel by 180°
        rot_kernel = np.rot90(np.rot90(np.array(self.kernels)))  # the kernel when applied to the image appears exactly the same just rotated by 180°, so we counteract that

        # Perform the sliding window operation
        for height in range(self.output_shape[1]):
            for width in range(self.output_shape[2]):
                for channels in range(self.output_shape[3]):
                    # ISN'T SELF.INPUT BECAUSE IT IS 4 DIMENSIONAL AND IMAGE SHOULD BE 3D THIS HAS TO BE THE "BATCH" OF THE WINDOW WE WANT TO MULTIPLY WITH THE KERNEL
                    window = self.window_batch[:, height:height + rot_kernel.shape[0], width:width + rot_kernel.shape[1], channels:channels + rot_kernel.shape[2]]
                    self.output[rot_kernel.shape[2], height, width, channels] = np.sum(window * self.kernels)

        return self.output   # output.shape needs to be (num_kernels, height, width, channels (rgb = 3)). Need to fix for loop self.output_shape working

    def backward(self, output_gradient, learning_rate):
        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_shape)

        for i in range(self.depth):
            for j in range(self.input_depth):
                kernels_gradient[i, j] = signal.correlate2d(self.input[j], output_gradient[i], "valid")
                input_gradient[j] += signal.convolve2d(output_gradient[i], self.kernels[i, j], "full")

        self.kernels -= learning_rate * kernels_gradient
        self.biases -= learning_rate * output_gradient
        return input_gradient


# Used to reshape the output of the conv2d layer
# forward reshapes input to output format
# backward reshapes output to input shape
class LayerFlatten:
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, input):
        return np.reshape(input, self.output_shape)

    def backward(self, output_gradient, learning_rate):
        return np.reshape(output_gradient, self.input_shape)

# Explanation: https://www.youtube.com/watch?v=sSWCeZVIN8w

# Max Pooling just means take the max value from each batch
# Drops unnecessary parts and also takes less time because the image is smaller
class LayerMaxPooling2D:
    # You have a pooling and a stride size.
    # The pooling size is the different parts we cut the image into (ex: parts of 2x2).
    # The stride is the pixels we move these batches at a time (ex: stride of 2 means 2 pixels at a time cutting the image in half)
    def __init__(self, pool_size, stride):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, input_data):
        self.input = input_data
        batch_size, input_height, input_width, input_channels = input_data.shape

        pool_height, pool_width = self.pool_size
        output_height = (input_height - pool_height) // self.stride + 1
        output_width = (input_width - pool_width) // self.stride + 1

        self.output = np.zeros((batch_size, output_height, output_width, input_channels))

        # does it have to be 4d isn't it just 3d?
        for b in range(batch_size):
            for h in range(output_height):
                for w in range(output_width):
                    for c in range(input_channels):
                        window = input_data[b, h * self.stride:h * self.stride + pool_height, w * self.stride:w * self.stride + pool_width, c]
                        self.output[b, h, w, c] = np.max(window)

        return self.output

    def backward(self, grad_output, learning_rate):
        batch_size, output_height, output_width, input_channels = grad_output.shape

        grad_input = np.zeros_like(self.input)

        for b in range(batch_size):
            for i in range(output_height):
                for j in range(output_width):
                    for c in range(input_channels):
                        window = self.input[b, i * self.stride:i * self.stride + self.pool_size[0], j * self.stride:j * self.stride + self.pool_size[1], c]
                        mask = (window == np.max(window))
                        grad_input[b, i * self.stride:i * self.stride + self.pool_size[0], j * self.stride:j * self.stride + self.pool_size[1], c] += grad_output[b, i, j, c] * mask

        return grad_input

# Min Pooling just means take the min value from each batch
# Used to get less important features of an image (low sharp values)
# You have a pooling and a stride size.
# The pooling size is the different parts we cut the image into (ex: parts of 2x2).
# The stride is the pixels we move these batches at a time (ex: stride of 2 means 2 pixels at a time cutting the image in half)
class LayerMinPooling2D:
    def __init__(self, pool_size, stride):
        self.pool_size = pool_size
        self.stride = stride
        self.input = None

    def forward(self, input_data):
        self.input = input_data
        batch_size, input_height, input_width, input_channels = input_data.shape

        pool_height, pool_width = self.pool_size
        output_height = (input_height - pool_height) // self.stride + 1
        output_width = (input_width - pool_width) // self.stride + 1

        output_data = np.zeros((batch_size, output_height, output_width, input_channels))

        for b in range(batch_size):
            for i in range(output_height):
                for j in range(output_width):
                    for c in range(input_channels):
                        window = input_data[b, i * self.stride:i * self.stride + pool_height, j * self.stride:j * self.stride + pool_width, c]
                        output_data[b, i, j, c] = np.min(window)

        return output_data

    def backward(self, grad_output):
        batch_size, output_height, output_width, input_channels = grad_output.shape

        grad_input = np.zeros_like(self.input)

        for b in range(batch_size):
            for i in range(output_height):
                for j in range(output_width):
                    for c in range(input_channels):
                        window = self.input[b, i * self.stride:i * self.stride + self.pool_size[0], j * self.stride:j * self.stride + self.pool_size[1], c]
                        mask = (window == np.min(window))
                        grad_input[b, i * self.stride:i * self.stride + self.pool_size[0], j * self.stride:j * self.stride + self.pool_size[1], c] += grad_output[b, i, j, c] * mask

        return grad_input


# Avg Pooling just means take the avg value from each batch
# Used to get the smooth features of an image
# You have a pooling and a stride size.
# The pooling size is the different parts we cut the image into (ex: parts of 2x2).
# The stride is the pixels we move these batches at a time (ex: stride of 2 means 2 pixels at a time cutting the image in half)
class LayerAvgPooling2D:
    def __init__(self, pool_size, stride):
        self.pool_size = pool_size
        self.stride = stride
        self.input = None

    def forward(self, input_data):
        self.input = input_data
        batch_size, input_height, input_width, input_channels = input_data.shape

        pool_height, pool_width = self.pool_size
        output_height = (input_height - pool_height) // self.stride + 1
        output_width = (input_width - pool_width) // self.stride + 1

        output_data = np.zeros((batch_size, output_height, output_width, input_channels))

        for b in range(batch_size):
            for i in range(output_height):
                for j in range(output_width):
                    for c in range(input_channels):
                        window = input_data[b, i * self.stride:i * self.stride + pool_height, j * self.stride:j * self.stride + pool_width, c]
                        output_data[b, i, j, c] = np.mean(window)

        return output_data

    def backward(self, grad_output):
        batch_size, output_height, output_width, input_channels = grad_output.shape

        grad_input = np.zeros_like(self.input)

        for b in range(batch_size):
            for i in range(output_height):
                for j in range(output_width):
                    for c in range(input_channels):
                        window = self.input[b, i * self.stride:i * self.stride + self.pool_size[0], j * self.stride:j * self.stride + self.pool_size[1], c]
                        grad_input[b, i * self.stride:i * self.stride + self.pool_size[0], j * self.stride:j * self.stride + self.pool_size[1], c] += grad_output[b, i, j, c] / (self.pool_size[0] * self.pool_size[1])

        return grad_input
