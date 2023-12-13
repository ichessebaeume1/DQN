import datetime

from activation_functions import ActivationReLU, ActivationSoftmax
from optimizer import OptimizerAdam
from loss import LossCategoricalCrossentropy
from accuracy import AccuracyCategorical
from layers import LayerHidden
from nn import NN
from dataset import *

# --- REQUIRED IS A DATASET EITHER IN EXTRACTED FORM, WITH A TRAIN AND A TEST FOLDER IN THEM EACH CONTAINING THE SAME AMOUNT OF VALID DATA FOR THE PURPOSE OF THE MODEL ---
# Download a Dataset
# download_dataset(URL, FILE_NAME, del_zip=delete_folder_after_unzip)

print("Preprocessing and Loading Dataset...")

# Create dataset from given data
X, y, X_test, y_test = mnist_preprocessing(FOLDER_NAME)

if not load_checkpoint:
    # Initialise the Model
    model = NN()

    # Add Layers and Activation Functions
    model.add(LayerHidden(X.shape[1], neurons))
    model.add(ActivationReLU())

    for layers in range(layers - 2):
        model.add(LayerHidden(neurons, neurons))
        model.add(ActivationReLU())

    model.add(LayerHidden(neurons, output_size))
    model.add(ActivationSoftmax())

    # Set the Loss Function, Optimizer Class, and Accuracy Calculation Object
    model.set_loss_opt_acc(loss=LossCategoricalCrossentropy(),
                           optimizer=OptimizerAdam(learning_rate=learning_rate, decay=decay),
                           accuracy=AccuracyCategorical())

    # Finalize the Model by going through its layers and adding final touches
    model.finalize()

    # TRAIN THE MODEEEEEEL !!!!!
    model.train(X, y, gens=generations, print_every=print_every, batch_size=batch_size, validation_data=(X_test, y_test))

else:
    model = NN.load(checkpoint_file_name)

model.evaluate(X_test, y_test, batch_size=batch_size)

model.save(f"model-checkpoint-{datetime.datetime.now().day}-{datetime.datetime.now().month}")

image_data = cv2.imread('tshirt.png', cv2.IMREAD_GRAYSCALE)
image_data = cv2.resize(image_data, (28, 28))
image_data = 255 - image_data
image_data = (image_data.reshape(1, -1).astype(np.float32) - 127.5) / 127.5

# Predict on the image
confidences = model.predict(image_data)
# Get prediction instead of confidence levels
predictions = model.output_layer_activation.predictions(confidences)
# Get label name from label index
prediction = predictions[0]
print(prediction)

# TODO:
# Dropout Layer active?
# Different training data
# Prediction and Inference
# Variations Meaning
# Initialise and train model class/function
# Config beautiful
