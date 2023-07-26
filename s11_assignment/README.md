# CIFAR10 -Advanced Training and Grad-Cam

## Objective 
The assignment requires building a specific training structure for training ResNet18 on the CIFAR10 dataset for 20 epochs. The structure involves creating a models folder with ResNet18 and ResNet34 models, a main.py file for training and testing loops, data splitting, epochs, batch size, optimizer, and scheduler configurations. Additionally, a utils.py file (or folder) is to be created for utilities like image transforms, GradCam, misclassification code, tensorboard, and advanced training policies.

## Code Structure

### [main.py](https://github.com/adil22jaleel/erav1-main-artefacts/blob/main/main.py)

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

### [utils.py](https://github.com/adil22jaleel/erav1-main-artefacts/blob/main/utils.py)

The utils.py file contains various utility functions and classes used for data preprocessing, visualization, and model analysis. Here's an explanation of each component:

- album_Compose_train class and album_Compose_test class:

These classes define data augmentation transformations using the Albumentations library for training and testing datasets, respectively.
The transformations include normalization, padding, random cropping, cutout (for training), and converting images to tensors.

- dataset_cifar10 class:

This class is used to load the CIFAR-10 dataset and define data loaders for training and testing.
It allows setting data augmentation for training and normalization for testing.
It also provides methods to compute data summary statistics (mean and standard deviation) for the entire dataset and visualize sample images from the dataset.
The different data augmentations applied are
    - Padding of 4 
    - RandomCrop of 32 x 32
    - Cutout of 8 & 8

![normalisedimages](https://github.com/adil22jaleel/era-v1-assignments/blob/main/s11_assignment/images/normalized image.png)

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

### [resnet.py](https://github.com/adil22jaleel/erav1-main-artefacts/blob/main/models/resnet.py)

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

### [s11.ipynb](https://github.com/adil22jaleel/era-v1-assignments/blob/main/s11_assignment/s11.ipynb)

The following is the workflow in which the model is trained and inferences obtained

1. Import Libraries
2. Load the training and testing dataset with Augmentations applied
3. Call the ResNet18 Network
4. Find the best LR using LR Finder library
5. Train the model
6. Plot the train and test accuracies and losses
7. Plot the misclassified images
8. Plot the gradcam for misclassified images
   
## One Cycle Policy

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
![lr_finder](https://github.com/adil22jaleel/era-v1-assignments/blob/main/s11_assignment/images/one%20cycle%20policy.png)
 

## Train and Test Metrics
The ResNet18 model was trained on the CIFAR10 dataset for 20 epochs.

The training and test loss curves are shown in the following image:

![train_test_metrics](https://github.com/adil22jaleel/era-v1-assignments/blob/main/s11_assignment/images/train_test_plots.png)

## Misclassified Images
A gallery of 10 misclassified images is shown below:

![misclassified_images](https://github.com/adil22jaleel/era-v1-assignments/blob/main/s11_assignment/images/misclassified.png)

## Grad Cam Images
The GradCam output on 10 misclassified images is shown below:

![gradcam_images](https://github.com/adil22jaleel/era-v1-assignments/blob/main/s11_assignment/images/gradcam_3layer.png)

```
Total number of Epochs: 20
Final Train Accuracy: 89.91%
Final Test Accuracy: 91.03%
```
```
EPOCH: 1 (LR: 6.2802914418342525e-06)

Batch_id=97: 100%|██████████| 98/98 [00:43<00:00,  2.26it/s]


Train Accuracy: 34.2100%
Train Average Loss: 1.80


Test Average loss: 1.3791
Test Accuracy: 49.68%


EPOCH: 2 (LR: 0.00013088435599945374)

Batch_id=97: 100%|██████████| 98/98 [00:42<00:00,  2.32it/s]


Train Accuracy: 51.8040%
Train Average Loss: 1.32


Test Average loss: 1.6631
Test Accuracy: 42.77%


EPOCH: 3 (LR: 0.0002554884205570732)

Batch_id=97: 100%|██████████| 98/98 [00:42<00:00,  2.30it/s]


Train Accuracy: 59.7800%
Train Average Loss: 1.12


Test Average loss: 1.2086
Test Accuracy: 57.99%


EPOCH: 4 (LR: 0.00038009248511469267)

Batch_id=97: 100%|██████████| 98/98 [00:42<00:00,  2.31it/s]


Train Accuracy: 64.5880%
Train Average Loss: 1.00


Test Average loss: 1.2801
Test Accuracy: 57.51%


EPOCH: 5 (LR: 0.0005046965496723121)

Batch_id=97: 100%|██████████| 98/98 [00:42<00:00,  2.30it/s]


Train Accuracy: 66.7240%
Train Average Loss: 0.94


Test Average loss: 1.8323
Test Accuracy: 45.24%


EPOCH: 6 (LR: 0.0006276019561961675)

Batch_id=97: 100%|██████████| 98/98 [00:42<00:00,  2.28it/s]


Train Accuracy: 69.3860%
Train Average Loss: 0.87


Test Average loss: 2.2849
Test Accuracy: 38.32%


EPOCH: 7 (LR: 0.0005857375334449004)

Batch_id=97: 100%|██████████| 98/98 [00:42<00:00,  2.30it/s]


Train Accuracy: 71.5420%
Train Average Loss: 0.82


Test Average loss: 1.0727
Test Accuracy: 63.16%


EPOCH: 8 (LR: 0.0005438731106936332)

Batch_id=97: 100%|██████████| 98/98 [00:42<00:00,  2.31it/s]


Train Accuracy: 73.3420%
Train Average Loss: 0.77


Test Average loss: 0.8883
Test Accuracy: 69.61%


EPOCH: 9 (LR: 0.000502008687942366)

Batch_id=97: 100%|██████████| 98/98 [00:42<00:00,  2.31it/s]


Train Accuracy: 74.8700%
Train Average Loss: 0.72


Test Average loss: 0.8629
Test Accuracy: 72.16%


EPOCH: 10 (LR: 0.00046014426519109896)

Batch_id=97: 100%|██████████| 98/98 [00:42<00:00,  2.30it/s]


Train Accuracy: 75.9440%
Train Average Loss: 0.70


Test Average loss: 0.9622
Test Accuracy: 69.91%


EPOCH: 11 (LR: 0.0004182798424398318)

Batch_id=97: 100%|██████████| 98/98 [00:42<00:00,  2.30it/s]


Train Accuracy: 77.3420%
Train Average Loss: 0.66


Test Average loss: 0.8596
Test Accuracy: 71.83%


EPOCH: 12 (LR: 0.00037641541968856467)

Batch_id=97: 100%|██████████| 98/98 [00:42<00:00,  2.31it/s]


Train Accuracy: 78.6620%
Train Average Loss: 0.62


Test Average loss: 0.9456
Test Accuracy: 69.58%


EPOCH: 13 (LR: 0.00033455099693729755)

Batch_id=97: 100%|██████████| 98/98 [00:42<00:00,  2.29it/s]


Train Accuracy: 79.5420%
Train Average Loss: 0.59


Test Average loss: 0.7644
Test Accuracy: 73.69%


EPOCH: 14 (LR: 0.0002926865741860304)

Batch_id=97: 100%|██████████| 98/98 [00:42<00:00,  2.30it/s]


Train Accuracy: 80.7600%
Train Average Loss: 0.56


Test Average loss: 0.7805
Test Accuracy: 74.75%


EPOCH: 15 (LR: 0.00025082215143476327)

Batch_id=97: 100%|██████████| 98/98 [00:42<00:00,  2.31it/s]


Train Accuracy: 81.7240%
Train Average Loss: 0.52


Test Average loss: 0.5662
Test Accuracy: 81.08%


EPOCH: 16 (LR: 0.00020895772868349615)

Batch_id=97: 100%|██████████| 98/98 [00:42<00:00,  2.30it/s]


Train Accuracy: 83.2480%
Train Average Loss: 0.49


Test Average loss: 0.5204
Test Accuracy: 83.08%


EPOCH: 17 (LR: 0.00016709330593222903)

Batch_id=97: 100%|██████████| 98/98 [00:42<00:00,  2.30it/s]


Train Accuracy: 85.0800%
Train Average Loss: 0.44


Test Average loss: 0.4388
Test Accuracy: 84.85%


EPOCH: 18 (LR: 0.00012522888318096186)

Batch_id=97: 100%|██████████| 98/98 [00:42<00:00,  2.30it/s]


Train Accuracy: 86.6200%
Train Average Loss: 0.39


Test Average loss: 0.4453
Test Accuracy: 85.52%


EPOCH: 19 (LR: 8.336446042969475e-05)

Batch_id=97: 100%|██████████| 98/98 [00:42<00:00,  2.31it/s]


Train Accuracy: 88.3280%
Train Average Loss: 0.34


Test Average loss: 0.3263
Test Accuracy: 89.15%


EPOCH: 20 (LR: 4.150003767842763e-05)

Batch_id=97: 100%|██████████| 98/98 [00:42<00:00,  2.29it/s]


Train Accuracy: 89.9140%
Train Average Loss: 0.29


Test Average loss: 0.2735
Test Accuracy: 91.03%
```
