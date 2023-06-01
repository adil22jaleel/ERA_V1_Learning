# MNIST Torch Model

A popular benchmark dataset in the fields of computer vision and machine learning is the MNIST dataset. It is made up of a number of grayscale pictures that represent the digits 0 through 9. The collection contains square images that are each 28x28 pixels in size, for a total of 784 pixels per image.

The torchvision module in PyTorch makes it simple to access well-liked datasets, such as MNIST, and makes the MNIST dataset available. A training set and a test set are the two primary segments of the dataset.

The test set has 10,000 images, whereas the training set has 60,000 images. The images have been separated into sections and are labelled with the corresponding digits.The goal of using the MNIST dataset in a neural network is to train a model that can accurately classify or recognize the handwritten digits in unseen images.


## Files in the project

The MNIST Torch Model conatins 3 main files.

1. utils.py
2. model.py 
3. S5.ipynb 

### utils.py

The utils.py file contains utility functions that are used in the project for common tasks such as data preprocessing and loading. Here's a brief description of the functions in utils.py:

⋅⋅* transforms_data(): This function loads and prepares the MNIST dataset for training and testing. It uses the torchvision library to download and preprocess the data, and returns data loaders that can be used to iterate over the data in batches during training and testing.


### model.py

The MNIST Net model, a convolutional neural network (CNN) architecture for categorising the MNIST handwritten digits dataset, is defined in the model.py file. Here is a quick summary of what model.py contains:

..*MNISTNet class: This class outlines the MNISTNet model's architecture. Convolutional layers (nn.Conv2d), pooling layers (F.max_pool2d), and fully linked layers (nn.Linear) are among the layers that make up this system. The model's forward pass is defined by the forward method.

### S5.ipynb

The notebook S5.ipynb file is a Jupyter Notebook that demonstrates how to train and test the MNIST Net model using the provided functions and utility modules. Here's a brief description of the contents of S5.ipynb:

Importing necessary modules and defining hyperparameters: The notebook starts with importing the required modules and specifying hyperparameters such as batch size, learning rate, and number of epochs.

Loading the data: The notebook uses the get_data_loaders function from utils.py to load the MNIST dataset and create data loaders for training and testing.

Defining the model, loss function, and optimizer: The notebook defines the MNISTNet model using the MNISTNet class from model.py. It also specifies the loss function Negative Loss Likelihood(nn.nll) and the optimizer (optim.SGD).

Training loop: The notebook contains a training loop that iterates over the data batches and performs forward and backward passes to train the model. It uses the train function defined in the notebook to handle the training process.

Testing loop: After the training loop, the notebook contains a testing loop that evaluates the trained model on the test dataset. It uses the test function defined in the notebook to handle the testing process.

Printing and analyzing results: Finally, the notebook prints the training and testing metrics, such as loss and accuracy.

## Usage

To use this project, follow these steps:

1. Make sure you have PyTorch and torchvision installed in your Python environment.

2. Place the utils.py, model.py, and S5.ipynb files in the same directory.

3. Open the S5.ipynb file in Jupyter Notebook or a compatible environment.

4. Run each cell in the notebook sequentially, following the instructions and comments provided.

5. Observe the training and testing progress, as well as the final metrics.

6. You can modify the hyperparameters, model architecture, or any other aspects of the code to suit your specific needs.
