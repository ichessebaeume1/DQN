import numpy as np

from activation_functions import ActivationSLCC, ActivationSoftmax
from loss import LossCategoricalCrossentropy
from layers import LayerInput, LayerHidden
import pickle
import visdomPlotting
import copy

class NN:
    def __init__(self):
        # Create a list of the layers inside the model
        self.layers = []
        self.softmax_classifier_output = None

    # Add layers to the model
    def add(self, layer):
        self.layers.append(layer)

    def set_loss_opt_acc(self, *, loss=None, optimizer=None, accuracy=None):
        if loss is not None:
            self.loss = loss
        if optimizer is not None:
            self.optimizer = optimizer
        if accuracy is not None:
            self.accuracy = accuracy

    def finalize(self):
        # Create an Input Layer
        self.input_layer = LayerInput()

        # Count all layers
        layer_count = len(self.layers)

        self.trainable_layers = []

        # Setting the properties for the layers
        for i in range(layer_count):
            # If it's the first hidden layer the previous one is the input layer and the next one is the next one
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i + 1]

            # If it's somewhere in between the previous layer is the one before it and the next one is the one after
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.layers[i + 1]

            # If it's the last hidden layer the previous one is the one before it and the next one is the output
            else:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]

            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])

        # Update loss object with trainable layers
        if self.loss is not None:
            self.loss.remember_trainable_layers(self.trainable_layers)

        # If output activation is Softmax and loss function is Categorical Cross-Entropy create an object of combined activation and loss function containing faster gradient calculation
        if isinstance(self.layers[-1], ActivationSoftmax) and isinstance(self.loss, LossCategoricalCrossentropy):
            # Create an object of combined activation and loss functions
            self.softmax_classifier_output = ActivationSLCC()

    def forward(self, X, training):
        # Call forward method on the input layer this will set the output property that the first layer in the for loop of finalize is expecting
        self.input_layer.forward(X, training)

        # Call the forward method of every layer in a chain and pass the output of the previous layer as the input
        for layer in self.layers:
            layer.forward(layer.prev.output)

        # Because "layer" is now the last layer from the list, return its output (which is the one of the final layer)
        return layer.output

    def backward(self, output, y):
        # If softmax classifier
        if self.softmax_classifier_output is not None:
            # First call backward method on the combined activation/loss this will set dinputs property
            self.softmax_classifier_output.backward(output, y)
            # Since we'll not call backward method of the last layer which is Softmax activation as we used combined activation/loss object, let's set dinputs in this object
            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs
            # Call backward method going through all the objects but last in reversed order passing dinputs as a parameter
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)

            return

        self.loss.backward(output, y)

        # Call backward method going through all the objects in reversed order passing dinputs as a parameter
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)

    def train(self, X, Y, *, gens=1, print_every=1, validation_data=None, batch_size=None):

        plotter = visdomPlotting.VisdomLinePlotter(env_name='NN_Model')

        # Initialize accuracy object
        self.accuracy.init(Y)

        # Default value if batch size is not being set
        train_steps = 1

        if validation_data is not None:
            # For better readability
            X_val, Y_val = validation_data

        # Calculate number of steps
        if batch_size is not None:
            train_steps = len(X) // batch_size
            # Dividing rounds down. If there are some remaining data, but not a full batch, this won't include it. Add 1 to include this not full batch
            if train_steps * batch_size < len(X):
                train_steps += 1

            if validation_data is not None:
                validation_steps = len(X_val) // batch_size
                # Dividing rounds down. If there are some remaining data, but nor full batch, this won't include it. Add 1 to include this not full batch
                if validation_steps * batch_size < len(X_val):
                    validation_steps += 1

        # Loop through each gen (1 gen = 1 loop in the network (with the entire data))
        for gen in range(1, gens + 1):
            # Print epoch number
            print(f'gen: {gen}')

            # Reset accumulated values in loss and accuracy objects
            self.loss.new_pass()
            self.accuracy.new_pass()

            # Iterate over steps (sending all the batches through the network)
            for step in range(train_steps):
                # If batch size is not set train using one step and full dataset
                if batch_size is None:
                    batch_X = X
                    batch_y = Y

                # Otherwise slice a batch
                else:
                    batch_X = X[step * batch_size:(step + 1) * batch_size]
                    batch_y = Y[step * batch_size:(step + 1) * batch_size]

                # Perform the forward pass
                output = self.forward(batch_X, training=True)

                # Calculate the loss
                data_loss, regularization_loss = self.loss.calculate(output, batch_y, include_regularization=True)
                loss = data_loss + regularization_loss

                # Calculate the accuracy
                predictions = self.output_layer_activation.predictions(output)
                acc = self.accuracy.calculate(predictions, batch_y)

                self.backward(output, batch_y)

                # Optimize (update parameters)
                self.optimizer.pre_update_params()

                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer)

                self.optimizer.post_update_params()

                # Print out data after a given number of steps
                if not step % print_every or step == train_steps - 1:
                    print(f'step: {step}, acc: {acc * 100:.3f}%, loss: {loss:.3f}, data_loss: {data_loss:.3f}, reg_loss: {regularization_loss:.3f}, lr: {self.optimizer.current_learning_rate}')

            # Get and print epoch loss and accuracy
            epoch_data_loss, epoch_regularization_loss = self.loss.calculate_accumulated(include_regularization=True)
            epoch_loss = epoch_data_loss + epoch_regularization_loss
            epoch_accuracy = self.accuracy.calculate_accumulated()

            # Print out data after a given number of steps
            print(f'Generation Statistic acc: {epoch_accuracy:.3f}, loss: {epoch_loss:.3f} data_loss: {epoch_data_loss:.3f}, reg_loss: {epoch_regularization_loss:.3f}, lr: {self.optimizer.current_learning_rate}')

            # Plot data if the config says you should
            plotter.plot("gen loss", "gen loss", "Generation Loss", gen, epoch_loss)
            plotter.plot('gen acc', 'gen acc', 'Generation Accuracy', gen, epoch_accuracy)
            plotter.plot('gen data_loss', 'gen data_loss', 'Generation Data Loss', gen, epoch_data_loss)
            plotter.plot('gen reg_loss', 'gen reg_loss', 'Generation Regularization Loss', gen, epoch_regularization_loss)

        if validation_data is not None:
            self.evaluate(*validation_data, batch_size=batch_size)

    def evaluate(self, X_val, Y_val, *, batch_size=None):
        # Default value if batch size is not being set
        validation_steps = 1
        # Calculate number of steps
        if batch_size is not None:
            validation_steps = len(X_val) // batch_size

            if validation_steps * batch_size < len(X_val):
                validation_steps += 1

        # Reset accumulated values in loss and accuracy objects
        self.loss.new_pass()
        self.accuracy.new_pass()

        # Iterate over steps
        for step in range(validation_steps):
            # If batch size is not set train using one step and full dataset
            if batch_size is None:
                batch_X = X_val
                batch_y = Y_val

            # Otherwise slice a batch
            else:
                batch_X = X_val[step * batch_size:(step + 1) * batch_size]
                batch_y = Y_val[step * batch_size:(step + 1) * batch_size]

            # Perform the forward pass
            output = self.forward(batch_X, training=False)

            # Calculate the loss
            self.loss.calculate(output, batch_y)

            # Get predictions and calculate an accuracy
            predictions = self.output_layer_activation.predictions(output)
            self.accuracy.calculate(predictions, batch_y)

        # Get and print validation loss and accuracy
        validation_loss = self.loss.calculate_accumulated()
        validation_accuracy = self.accuracy.calculate_accumulated()

        print(f'validation, acc: {validation_accuracy:.3f}, loss: {validation_loss:.3f}')

    def get_parameters(self):
        # Create a list for parameters
        parameters = []
        # Iterable trainable layers and get their parameters
        for layer in self.trainable_layers:
            parameters.append(layer.get_parameters())

        return parameters

    # Updates the model with new parameters
    def set_parameters(self, parameters):
        # Iterate over the parameters and layers and update each layers with each set of the parameters
        for parameter_set, layer in zip(parameters, self.trainable_layers):
            layer.set_parameters(*parameter_set)

    def save_parameters(self, path):
        with open(path, "wb") as file:
            pickle.dump(self.get_parameters(), file)

    def load_parameters(self, path):
        with open(path, "rb") as file:
            self.set_parameters(pickle.load(file))

    def save(self, path):
        model = copy.deepcopy(self)

        # Reset accumulated values in loss and accuracy objects
        model.loss.new_pass()
        model.accuracy.new_pass()

        # Remove data from input layer and gradients from the loss object
        model.input_layer.__dict__.pop('output', None)
        model.loss.__dict__.pop('dinputs', None)

        # For each layer remove inputs, output and dinputs properties
        for layer in model.layers:
            for property in ['inputs', 'output', 'dinputs', 'dweights', 'dbiases']:
                layer.__dict__.pop(property, None)

        # Open a file in the binary-write mode and save the model
        with open(path, 'wb') as f:
            pickle.dump(model, f)

    @staticmethod
    def load(path):
        # Open file in the binary-read mode, load a model
        with open(path, 'rb') as file:
            model = pickle.load(file)
        # Return a model
        return model

    def predict(self, X, *, batch_size=None):   # Basically just do what we have done before with the generations just now do it with one sample (send it through the network once and read the outputs)
        # Default value if batch size is not being set
        prediction_steps = 1
        # Calculate number of steps
        if batch_size is not None:
            prediction_steps = len(X) // batch_size

            if prediction_steps * batch_size < len(X):
                prediction_steps += 1

        output = []

        # Iterate over steps
        for step in range(prediction_steps):
            # If batch size is not set - train using one step and full dataset
            if batch_size is None:
                batch_X = X
            # Otherwise slice a batch
            else:
                batch_X = X[step * batch_size:(step + 1) * batch_size]
            # Perform the forward pass
            batch_output = self.forward(batch_X, training=False)
            # Append batch prediction to the list of predictions
            output.append(batch_output)

        return np.vstack(output)
