import numpy as np

class Accuracy:
    def calculate(self, pred, y):
        # Compare y (the true values) with pred (the predicted values) and return accuracy as a mean
        comparison = self.compare(pred, y)
        accuracy = np.mean(comparison)

        # Add accumulated sum of matching values and sample count
        self.accumulated_sum += np.sum(comparison)
        self.accumulated_count += len(comparison)

        return accuracy

    # Calculates accumulated accuracy
    def calculate_accumulated(self):
        # Calculate an accuracy
        accuracy = self.accumulated_sum / self.accumulated_count
        # Return the data and regularization losses
        return accuracy

    # Reset variables for accumulated accuracy
    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0

# Accuracy calculation for classification model
class AccuracyCategorical(Accuracy):
    def __init__(self, *, binary=False):
        # Binary mode?
        self.binary = binary

    # No initialization is needed
    def init(self, y):
        pass

    # Compares predictions to the ground truth values
    def compare(self, predictions, y):
        if not self.binary and len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        return predictions == y
