# CIFAR10 -Advanced Training and Grad-Cam

## Objective 
The assignment requires building a specific training structure for training ResNet18 on the CIFAR10 dataset for 20 epochs. The structure involves creating a models folder with ResNet18 and ResNet34 models, a main.py file for training and testing loops, data splitting, epochs, batch size, optimizer, and scheduler configurations. Additionally, a utils.py file (or folder) is to be created for utilities like image transforms, GradCam, misclassification code, tensorboard, and advanced training policies.

## Code Structure

### [main.py]()

The main.py file provided in the assignment is a Python script used to train and test models using PyTorch. It contains several functions and a fitting function to facilitate the training process. Here's an explanation of each function:

- train(model, device, train_loader, optimizer, scheduler): This function performs the training loop for the given model on the provided training dataset. It takes the following arguments:

    - model: The PyTorch model to be trained.

    - device: Specifies whether to use "cpu" or "cuda" (GPU) for training.
    - train_loader: Torch DataLoader for the training dataset.
    - optimizer: The optimizer to be used for updating the model's parameters during training.
    - scheduler: The learning rate scheduler to adjust the learning rate during training.

The function iterates through the training data, calculates the loss, performs backpropagation, and updates the model's parameters. It also keeps track of training loss and accuracy and returns them.

- test(model, device, test_loader): This function performs the testing loop for the given model on the provided test dataset. It takes the following arguments:

    - model: The PyTorch model to be tested.
    - device: Specifies whether to use "cpu" or "cuda" (GPU) for testing.
    - test_loader: Torch DataLoader for the test dataset.

The function sets the model to evaluation mode (disabling dropout and batch normalization), iterates through the test data, calculates the test loss and accuracy, and prints the results.

- fit_model(net, device, train_loader, test_loader, optimizer, scheduler, NUM_EPOCHS=20): This function trains and tests the model using the train and test functions. It takes the following arguments:

    - net: The PyTorch model to be trained and tested.
    - device: Specifies whether to use "cpu" or "cuda" (GPU) for training and testing.
    - train_loader: Torch DataLoader for the training dataset with augmentations.
    - test_loader: Torch DataLoader for the test dataset with normalized images.
    - optimizer: The optimizer to be used for updating the model's parameters during training.
    - scheduler: The learning rate scheduler to adjust the learning rate during training.
    - NUM_EPOCHS: (optional) The number of epochs to train the model (default value is 20).

The function iterates through the specified number of epochs, performing training and testing for each epoch and storing the training and testing accuracy and loss. It returns the trained model along with the recorded training and testing accuracy, loss, and learning rate history.

### [utils.py]()

The utils.py file contains various utility functions and classes used for data preprocessing, visualization, and model analysis. Here's an explanation of each component:

- album_Compose_train class and album_Compose_test class:

These classes define data augmentation transformations using the Albumentations library for training and testing datasets, respectively.
The transformations include normalization, padding, random cropping, cutout (for training), and converting images to tensors.

- dataset_cifar10 class:

This class is used to load the CIFAR-10 dataset and define data loaders for training and testing.
It allows setting data augmentation for training and normalization for testing.
It also provides methods to compute data summary statistics (mean and standard deviation) for the entire dataset and visualize sample images from the dataset.

- get_device function:

This function checks if CUDA (GPU) is available and returns the appropriate device ("cpu" or "cuda") to be used for training.

- unnormalize function:

This function is used to unnormalize images that were normalized using the provided mean and standard deviation.

- custom_lr_finder function:

This function uses the Learning Rate Finder technique to determine the optimal learning rate for the model.
It takes a model, criterion, optimizer, device, and data loader and returns the best learning rate found during the range test.

- plot_misclassified function:

This function is used to plot misclassified images from the test set along with their true and predicted labels.
It takes the model, test loader, classes (class names or class indices), device, and other parameters to visualize the misclassified images.

- train_test_plots function:

This function is used to plot the training and testing loss curves and accuracy curves over epochs.
It takes the training history as input, which contains training and testing accuracy and loss values.

- gradcam_vis function:

This function is used to visualize Grad-CAM (Gradient-weighted Class Activation Mapping) outputs for misclassified images.
It takes the model, target layers for Grad-CAM, misclassified images, classes, and plot size as inputs to visualize the Grad-CAM outputs.
These utility functions and classes facilitate data preprocessing, analysis, and visualization, making it easier to train and evaluate models on the CIFAR-10 dataset.

### [model.py]()

The model.py file contains the implementation of the ResNet architecture in PyTorch. It defines two models: ResNet18() and ResNet34(), which are variations of the ResNet model with 18 and 34 layers, respectively.

- BasicBlock class:

    - This class defines the basic building block used in the ResNet architecture.
    - It consists of two 3x3 convolutional layers, each followed by batch normalization and ReLU activation.
    - The block may include a shortcut connection if the stride is not equal to 1 or if the number of input channels is different from the output channels (to match dimensions).
    - The forward method performs the forward pass through the basic block, including the shortcut connection and ReLU activation.

- ResNet class:

    - This class defines the ResNet model, which is composed of multiple stacked BasicBlock layers.
    - The constructor initializes the model with the number of classes (defaulted to 10 for CIFAR-10) and the initial number of input channels (64).
    - The model consists of a initial 3x3 convolutional layer followed by batch normalization and ReLU activation.
    - It then includes four stages, each containing a sequence of BasicBlock layers with different numbers of channels and strides.
    - The final fully connected layer reduces the features to the number of classes.

- ResNet18() and ResNet34() functions:

    - These functions create specific instances of the ResNet class with different numbers of layers.
    - ResNet18() creates a ResNet model with 18 layers (composed of 2 layers each in the four stages).
    - ResNet34() creates a ResNet model with 34 layers (composed of 3, 4, 6, and 3 layers in the four stages, respectively).
    - These models can be used as backbone architectures for various computer vision tasks, especially for image classification. For example, in the assignment, the task is to train ResNet18 on the CIFAR-10 dataset for 20 epochs using the provided training structure in the main.py file.

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 64, 32, 32]           1,728
       BatchNorm2d-2           [-1, 64, 32, 32]             128
            Conv2d-3           [-1, 64, 32, 32]          36,864
       BatchNorm2d-4           [-1, 64, 32, 32]             128
            Conv2d-5           [-1, 64, 32, 32]          36,864
       BatchNorm2d-6           [-1, 64, 32, 32]             128
        BasicBlock-7           [-1, 64, 32, 32]               0
            Conv2d-8           [-1, 64, 32, 32]          36,864
       BatchNorm2d-9           [-1, 64, 32, 32]             128
           Conv2d-10           [-1, 64, 32, 32]          36,864
      BatchNorm2d-11           [-1, 64, 32, 32]             128
       BasicBlock-12           [-1, 64, 32, 32]               0
           Conv2d-13          [-1, 128, 16, 16]          73,728
      BatchNorm2d-14          [-1, 128, 16, 16]             256
           Conv2d-15          [-1, 128, 16, 16]         147,456
      BatchNorm2d-16          [-1, 128, 16, 16]             256
           Conv2d-17          [-1, 128, 16, 16]           8,192
      BatchNorm2d-18          [-1, 128, 16, 16]             256
       BasicBlock-19          [-1, 128, 16, 16]               0
           Conv2d-20          [-1, 128, 16, 16]         147,456
      BatchNorm2d-21          [-1, 128, 16, 16]             256
           Conv2d-22          [-1, 128, 16, 16]         147,456
      BatchNorm2d-23          [-1, 128, 16, 16]             256
       BasicBlock-24          [-1, 128, 16, 16]               0
           Conv2d-25            [-1, 256, 8, 8]         294,912
      BatchNorm2d-26            [-1, 256, 8, 8]             512
           Conv2d-27            [-1, 256, 8, 8]         589,824
      BatchNorm2d-28            [-1, 256, 8, 8]             512
           Conv2d-29            [-1, 256, 8, 8]          32,768
      BatchNorm2d-30            [-1, 256, 8, 8]             512
       BasicBlock-31            [-1, 256, 8, 8]               0
           Conv2d-32            [-1, 256, 8, 8]         589,824
      BatchNorm2d-33            [-1, 256, 8, 8]             512
           Conv2d-34            [-1, 256, 8, 8]         589,824
      BatchNorm2d-35            [-1, 256, 8, 8]             512
       BasicBlock-36            [-1, 256, 8, 8]               0
           Conv2d-37            [-1, 512, 4, 4]       1,179,648
      BatchNorm2d-38            [-1, 512, 4, 4]           1,024
           Conv2d-39            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-40            [-1, 512, 4, 4]           1,024
           Conv2d-41            [-1, 512, 4, 4]         131,072
      BatchNorm2d-42            [-1, 512, 4, 4]           1,024
       BasicBlock-43            [-1, 512, 4, 4]               0
           Conv2d-44            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-45            [-1, 512, 4, 4]           1,024
           Conv2d-46            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-47            [-1, 512, 4, 4]           1,024
       BasicBlock-48            [-1, 512, 4, 4]               0
           Linear-49                   [-1, 10]           5,130
================================================================
Total params: 11,173,962
Trainable params: 11,173,962
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 11.25
Params size (MB): 42.63
Estimated Total Size (MB): 53.89
----------------------------------------------------------------
```

### [s11.ipynb]()

The following is the workflow in which the model is trained and inferenced

1. Import Libraries
2. Load the training and testing dataset with Augmentations applied
3. Call the ResNet18 Network
4. Find the best LR using LR Finder library
5. Train the model
6. Plot the train and test accuracies and losses
7. Plot the misclassified images
8. Plot the gradcam for misclassified images 

## Train and Test Metrics
The ResNet18 model was trained on the CIFAR10 dataset for 20 epochs.

The training and test loss curves are shown in the following image:

![train_test_metrics]()

## Misclassified Images
A gallery of 10 misclassified images is shown below:

![misclassified_images]()

## Grad Cam Images
The GradCam output on 10 misclassified images is shown below:

![gradcam_images]()


## Training Log

Total number of Epochs: 20
Final Train Accuracy: 
Final Test Accuracy: 

```

```