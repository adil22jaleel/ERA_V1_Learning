#  Residual Connections in CNNs and One Cycle Policy!

## Objectives

1. Write a ResNet architecture for CIFAR10 that has the following architecture:
    - PrepLayer - Conv 3x3 s1, p1) >> BN >> RELU [64k]
    - Layer1 -
        - X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
        - R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k] 
        - Add(X, R1)
    - Layer 2 -
        - Conv 3x3 [256k]
        - MaxPooling2D
        - BN
        - ReLU
    - Layer 3 -
        - X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]
        - R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]
        - Add(X, R2)
    - MaxPooling with Kernel Size 4
    - FC Layer 
    - SoftMax
    - Uses One Cycle Policy such that:
        - Total Epochs = 24
        - Max at Epoch = 5
        - Find LRMIN and LRMAX and No Annihilation

    - Uses this transform -RandomCrop 32, 32 (after padding of 4) >> FlipLR >> Followed by CutOut(8, 8)
    - Batch size = 512
    - Use ADAM, and CrossEntropyLoss
    - Target Accuracy: 90%
 

## Code Structure

### [dataset.py](https://github.com/adil22jaleel/era-v1-assignments/blob/main/s10_assignment/main/dataset.py)

The dataset.py file provides classes and functions for loading and preprocessing the CIFAR-10 dataset using PyTorch and Albumentations.

####  Data Processing and Handling
- Initializes an instance of the class with the desired batch size for data loading.
- Contains all the loader functions for pytorch implementation 
- There are functions written for different requirements
    - data(self, train_flag): Returns the CIFAR-10 dataset with the specified transformations based on the train_flag parameter. If train_flag is True, data augmentation transformations are applied during training. If False, only normalization transformations are applied during testing.
    - loader(self, train_flag=True): Returns a PyTorch data loader for the CIFAR-10 dataset based on the train_flag parameter. The data loader is configured with the specified batch size and other parameters.
    - data_summary_stats(self): Computes and prints the summary statistics of the CIFAR-10 dataset, including the shape, mean, and standard deviation of the concatenated train and test data.
    - sample_pictures(self, train_flag=True, return_flag=False): Displays a grid of sample images from the CIFAR-10 dataset. If train_flag is True, samples are taken from the training set; otherwise, samples are taken from the test set. If return_flag is True, the sampled images and corresponding labels are returned as well.
    - unnormalize(img): Takes a normalized image tensor and returns the unnormalized image in numpy array format.

####  Image Augmentation 
- Augmentation is performed using the Albumentations library. 
- Albumentations is a popular open-source library for image augmentation in machine learning and computer vision tasks. It provides a wide range of augmentation techniques that can be applied to images, including geometric transformations, color manipulations, pixel-level operations, and more.
- Three techniques are applied in the training data loader: horizontal flipping, shiftScaleRotate, and coarseDropout.
- album_Compose_train class: Initializes an instance of the class with a composition of Albumentations transformations for data augmentation during training.
- album_Compose_test class:Initializes an instance of the class with a composition of Albumentations transformations for normalization during testing.
- The various augmentations we have done are
    - Normalize: Normalizes the image by subtracting the mean and dividing by the standard deviation.
    - PadIfNeeded: Pads the image if its height or width is smaller than the specified minimum height or width.
    - RandomCrop: Randomly crops the image to the specified height and width.
    - HorizontalFlip: Flips the image horizontally.
    - Cutout: Applies random cutout augmentation by removing rectangular regions of the image.
    - ToTensorV2: Converts the image to a PyTorch tensor.

The following image shows the image post augmentation.
![image_augmentation](https://github.com/adil22jaleel/era-v1-assignments/blob/main/s10_assignment/assets/post_normalizer.png)


### [custom_resnet.py](https://github.com/adil22jaleel/era-v1-assignments/blob/main/s10_assignment/main/custom_resnet.py)


#### Model Architecture
The CustomResNet model consists of several components:

- ResBlock: This class represents a residual block, which is a fundamental building block of the ResNet architecture.
- CustomResNet: The main model class that combines various layers and residual blocks to form the complete architecture.


##### Residual Block (ResBlock)
The ResBlock class defines a single residual block. It performs two sets of convolutional operations with batch normalization and ReLU activation. The architecture of each residual block can be summarized as follows:
```
-----------------------------
| Conv2d (3x3)               |
| BatchNorm2d                 |
| ReLU                        |
| Conv2d (3x3)               |
| BatchNorm2d                 |
| ReLU                        |
-----------------------------
```
##### CustomResNet Architecture

- preplayer: This initial layer applies a convolutional operation with batch normalization and ReLU activation to the input image. It converts the input from 3 channels to 64 channels.

- layer1_X: This layer performs a convolutional operation followed by max pooling, batch normalization, and ReLU activation. It converts the input from 64 channels to 128 channels and reduces the spatial dimensions by half.

- layer1_R: This is the first residual block. It takes the output of layer1_X as input and performs a series of convolutional operations with batch normalization and ReLU activation.

- layer2_X: This layer is similar to layer1_X but converts the input from 128 channels to 256 channels.

- layer3_X: This layer is similar to layer1_X but converts the input from 256 channels to 512 channels.

- layer3_R: This is the second residual block, similar to layer1_R, but takes the output of layer3_X as input.

- maxpool_4: This max pooling layer reduces the spatial dimensions of the input by a factor of 4.

- linear: This linear layer performs the final classification. It takes the flattened output of maxpool_4 and maps it to the output dimension of 10, representing the number of classes.

- forward method: This method defines the forward pass of the model. It sequentially passes the input through each layer and applies element-wise addition between the output of layer1_X and the output of the first residual block (R1). Similarly, it applies element-wise addition between the output of layer3_X and the output of the second residual block (R3). Finally, it performs a softmax activation on the output and returns the result.

![model_archi](https://github.com/adil22jaleel/era-v1-assignments/blob/main/s10_assignment/assets/model_architecture.png)



### [backpropgation.py](https://github.com/adil22jaleel/era-v1-assignments/blob/main/s10_assignment/main/backpropagation.py)



#### Training
The train function in the script is responsible for training the model. Here are the key steps involved:

- Set the model to training mode and initialize variables for tracking loss, accuracy, and the number of processed samples.
- Iterate over the training data using the train_loader.
- Move the data and target tensors to the device (e.g., GPU) if available.
- Zero the gradients of the optimizer.
- Forward pass: Pass the input data through the model to obtain predictions.
- Calculate the loss between the predictions and the target labels using the specified criterion.
- Backpropagation: Compute gradients of the loss with respect to the model parameters.
- Update the model parameters by calling optimizer.step().
- Update the tracked variables (loss, accuracy, processed samples) for reporting and visualization.
- Optionally update the learning rate using a learning rate scheduler.

#### Testing
The test function in the script is responsible for evaluating the model on the test data. Here are the key steps involved:

- Set the model to evaluation mode.
- Iterate over the test data using the test_loader.
- Move the data and target tensors to the device (e.g., GPU) if available.
- Forward pass: Pass the input data through the model to obtain predictions.
- Calculate the loss between the predictions and the target labels using the specified criterion.
- Update the tracked variables (loss, accuracy, processed samples) for reporting.

### [visualize.py](https://github.com/adil22jaleel/era-v1-assignments/blob/main/s10_assignment/main/visualize.py)

This script provides functions to visualize the misclassified images and plot the training and test statistics.

#### Functions

- plot_misclassified_images(model, test_loader, classes, device)
    - This function plots the misclassified images from the test dataset along with their actual and predicted labels.

- train_plot_stats(train_losses, train_accuracies)
    - This function plots the training loss and accuracy statistics.

- test_plot_stats(test_losses, test_accuracies)
    - This function plots the test loss and accuracy statistics.

### [utils.py](https://github.com/adil22jaleel/era-v1-assignments/blob/main/s10_assignment/main/utils.py)

This Python script defines a utility function for printing the summary of a PyTorch model using the torchsummary package. The function print_summary takes a model instance and an optional input_size parameter as input and displays a summary of the model's architecture.


### One Cycle Policy

The OneCycleLR learning rate scheduler in combination with the torch_lr_finder library provides a way to automatically find the optimal learning rate for training deep learning models. By using the optimal learning rate, you can improve the model's convergence and generalization. Understanding the code snippet and the usage of OneCycleLR enables you to apply this learning rate scheduling technique to your own deep learning projects.

- Define your criterion (loss function) and optimizer. We have added ADAM Optimizer and CrossEntropy Loss. ADAM is an extension of the stochastic gradient descent (SGD) algorithm that adapts the learning rate for each parameter based on the first and second moments of the gradients.

- Create an instance of LRFinder, passing the model, optimizer, criterion, and the device on which the computations will be performed:
The Learning Rate (LR) Finder is a technique commonly used in the field of deep learning to determine an optimal learning rate for a neural network during the training process. The learning rate is a hyperparameter that controls the step size at which the model's parameters are updated during optimization.

```python
lr_finder = LRFinder(model, optimizer, criterion, device=device)
```

- Run the learning rate range test using the range_test method. Specify the end learning rate, the number of iterations, and the step mode:

```python
lr_finder.range_test(trainloader, end_lr=10, num_iter=100, step_mode="exp")
```

- Plot the learning rate range test results and identify the optimal learning rate:

```python
_, best_lr = lr_finder.plot()
```
- Reset the LRFinder to ensure it's ready for the next step:

```python
lr_finder.reset()
```
The following diagram shows how the max LR is retrieved from the lr finder plot. 
![lr_finder](https://github.com/adil22jaleel/era-v1-assignments/blob/main/s10_assignment/assets/lr_finder.png)
## Inferences and Analysis



### Model Final Metrics

- Batch Size: 512
- Training accuracy: 98.12%
- Test accuracy: 93.08%
- Number of epochs model trained: 24
- Number of parameters: 6,573,130 parameters
- LR-MAX: 5.21E-04
- LR-MIN: 5.21E-05



### Training Log
```
EPOCH: 1
Batch_id=97: 100%|██████████| 98/98 [00:22<00:00,  4.35it/s]

Train Average Loss: 1.78%
Train Accuracy: 37.1700

Test Average loss: 1.2814
Test Accuracy: 53.96


EPOCH: 2
Batch_id=97: 100%|██████████| 98/98 [00:23<00:00,  4.19it/s]

Train Average Loss: 1.13%
Train Accuracy: 59.8360

Test Average loss: 0.9731
Test Accuracy: 65.88


EPOCH: 3
Batch_id=97: 100%|██████████| 98/98 [00:22<00:00,  4.37it/s]

Train Average Loss: 0.89%
Train Accuracy: 68.7680

Test Average loss: 0.9643
Test Accuracy: 66.28


EPOCH: 4
Batch_id=97: 100%|██████████| 98/98 [00:22<00:00,  4.39it/s]

Train Average Loss: 0.73%
Train Accuracy: 74.3460

Test Average loss: 0.7483
Test Accuracy: 74.51


EPOCH: 5
Batch_id=97: 100%|██████████| 98/98 [00:22<00:00,  4.39it/s]

Train Average Loss: 0.63%
Train Accuracy: 78.1660

Test Average loss: 0.6488
Test Accuracy: 77.92


EPOCH: 6
Batch_id=97: 100%|██████████| 98/98 [00:24<00:00,  3.94it/s]

Train Average Loss: 0.56%
Train Accuracy: 80.7300

Test Average loss: 0.7215
Test Accuracy: 76.01


EPOCH: 7
Batch_id=97: 100%|██████████| 98/98 [00:22<00:00,  4.33it/s]

Train Average Loss: 0.49%
Train Accuracy: 82.9000

Test Average loss: 0.5833
Test Accuracy: 80.02


EPOCH: 8
Batch_id=97: 100%|██████████| 98/98 [00:21<00:00,  4.46it/s]

Train Average Loss: 0.45%
Train Accuracy: 84.4880

Test Average loss: 0.4706
Test Accuracy: 83.38


EPOCH: 9
Batch_id=97: 100%|██████████| 98/98 [00:22<00:00,  4.43it/s]

Train Average Loss: 0.41%
Train Accuracy: 86.1700
Test Average loss: 0.5845
Test Accuracy: 80.01


EPOCH: 10
Batch_id=97: 100%|██████████| 98/98 [00:22<00:00,  4.37it/s]

Train Average Loss: 0.39%
Train Accuracy: 86.8640

Test Average loss: 0.5024
Test Accuracy: 83.09


EPOCH: 11
Batch_id=97: 100%|██████████| 98/98 [00:22<00:00,  4.37it/s]

Train Average Loss: 0.36%
Train Accuracy: 87.7200

Test Average loss: 0.4166
Test Accuracy: 85.55


EPOCH: 12
Batch_id=97: 100%|██████████| 98/98 [00:22<00:00,  4.39it/s]

Train Average Loss: 0.34%
Train Accuracy: 88.5540

Test Average loss: 0.4538
Test Accuracy: 84.29


EPOCH: 13
Batch_id=97: 100%|██████████| 98/98 [00:22<00:00,  4.36it/s]

Train Average Loss: 0.31%
Train Accuracy: 89.1900

Test Average loss: 0.4203
Test Accuracy: 85.81


EPOCH: 14
Batch_id=97: 100%|██████████| 98/98 [00:22<00:00,  4.34it/s]

Train Average Loss: 0.30%
Train Accuracy: 89.8540

Test Average loss: 0.4423
Test Accuracy: 84.81


EPOCH: 15
Batch_id=97: 100%|██████████| 98/98 [00:22<00:00,  4.40it/s]

Train Average Loss: 0.28%
Train Accuracy: 90.6760
Test Average loss: 0.4379
Test Accuracy: 85.25


EPOCH: 16
Batch_id=97: 100%|██████████| 98/98 [00:22<00:00,  4.32it/s]

Train Average Loss: 0.26%
Train Accuracy: 91.3680

Test Average loss: 0.3986
Test Accuracy: 86.65


EPOCH: 17
Batch_id=97: 100%|██████████| 98/98 [00:23<00:00,  4.26it/s]

Train Average Loss: 0.24%
Train Accuracy: 91.9820

Test Average loss: 0.3587
Test Accuracy: 88.01


EPOCH: 18
Batch_id=97: 100%|██████████| 98/98 [00:22<00:00,  4.27it/s]

Train Average Loss: 0.21%
Train Accuracy: 93.1420

Test Average loss: 0.3401
Test Accuracy: 88.42


EPOCH: 19
Batch_id=97: 100%|██████████| 98/98 [00:23<00:00,  4.22it/s]

Train Average Loss: 0.20%
Train Accuracy: 93.4880

Test Average loss: 0.2919
Test Accuracy: 90.22


EPOCH: 20
Batch_id=97: 100%|██████████| 98/98 [00:23<00:00,  4.22it/s]

Train Average Loss: 0.17%
Train Accuracy: 94.4720

Test Average loss: 0.2871
Test Accuracy: 90.39


EPOCH: 21
Batch_id=97: 100%|██████████| 98/98 [00:23<00:00,  4.22it/s]

Train Average Loss: 0.14%
Train Accuracy: 95.7020

Test Average loss: 0.2737
Test Accuracy: 91.08


EPOCH: 22
Batch_id=97: 100%|██████████| 98/98 [00:23<00:00,  4.26it/s]

Train Average Loss: 0.12%
Train Accuracy: 96.4440
Test Average loss: 0.2795
Test Accuracy: 90.78


EPOCH: 23
Batch_id=97: 100%|██████████| 98/98 [00:23<00:00,  4.14it/s]

Train Average Loss: 0.09%
Train Accuracy: 97.2980

Test Average loss: 0.2216
Test Accuracy: 92.83


EPOCH: 24
Batch_id=97: 100%|██████████| 98/98 [00:23<00:00,  4.17it/s]

Train Average Loss: 0.07%
Train Accuracy: 98.1240

Test Average loss: 0.2117
Test Accuracy: 93.08
```



### Misclassified Images
![misclassified_images](https://github.com/adil22jaleel/era-v1-assignments/blob/main/s10_assignment/assets/misclassified.png)
