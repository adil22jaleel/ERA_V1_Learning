# CNN on CIFAR Dataset with various Normalization Techniques
## Problem Statement
- Make a model using CIFAR dataset
    - Network with Group Normalization
    - Network with Layer Normalization
    - Network with Batch Normalization

- Find 10 misclassified images for each of the 3 models, and show them as a 5x2 image matrix in 3 separately annotated images.
- Comparison of accuracy and losses for the normalization techniques

## Solution
### Code Overview
This code aims to study the impact of various normalization techniques on a CNN model trained on the CIFAR dataset. It includes the following

#### [dataset.py](https://github.com/adil22jaleel/era-v1-assignments/blob/main/s8_assignment/dataset.py)

This code provides a function get_loader that helps in obtaining instances of train and test data loaders for the CIFAR-10 dataset. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. This dataset is commonly used for image classification tasks.

#### [model.py](https://github.com/adil22jaleel/era-v1-assignments/blob/main/s8_assignment/model.py)

This code file contains implementations of multiple models for various tasks, such as image classification, object detection, or generative tasks. Each model is implemented as a separate class, providing modularity and flexibility for different use cases. Below is an overview of the available models:

CNN model called cnn_norm that can be configured to use different normalization techniques such as batch normalization, layer normalization, or group normalization. The model consists of several convolutional blocks followed by pooling and fully connected layers.

Here's an overview of the model's structure:

The constructor (__init__) takes parameters to specify which normalization technique to use (use_batch_norm, use_layer_norm, use_group_norm). It also takes a parameter num_groups for group normalization.

- If batch normalization is selected (use_batch_norm=True), the model defines the layers for batch normalization after each convolutional layer. It starts with an input block (convblock1) that consists of a convolutional layer, ReLU activation, batch normalization, and dropout. Then, there are three convolutional blocks (convblock2, convblock4, convblock5) each followed by batch normalization, ReLU activation, and dropout. Two transition blocks (convblock3, convblock7) contain 1x1 convolutions. The model ends with a global average pooling layer (gap) and a 1x1 convolutional layer (convblock11) for classification.

- If layer normalization is selected (use_layer_norm=True), the model follows a similar structure as above, but replaces the batch normalization layers with layer normalization.

- If group normalization is selected (use_group_norm=True), the model replaces the batch normalization layers with group normalization. The number of groups is determined by the num_groups parameter.

- The purpose of using normalization techniques is to improve the training and generalization performance of the model by normalizing the activations within each layer. 

- Batch Normalization (BN): For each channel over each minibatch.

- Group Normalization (GN): For each group in split of channel over each image.

- Layer Normalization (LN): Over all channel for each image.

![bn_gn_ln](https://github.com/adil22jaleel/era-v1-assignments/blob/main/s8_assignment/images/bn_gn_ln.png)

#### [utils.py](https://github.com/adil22jaleel/era-v1-assignments/blob/main/s8_assignment/utils.py)

This Python script defines a utility function for printing the summary of a PyTorch model using the torchsummary package. The function print_summary takes a model instance and an optional input_size parameter as input and displays a summary of the model's architecture.

#### [backpropagation.py](https://github.com/adil22jaleel/era-v1-assignments/blob/main/s8_assignment/backpropagation.py)

The backpropagation.py script implements the backpropagation algorithm for training neural networks. It contains functions for training and testing a model using the backpropagation algorithm.

The backpropagation.py script provides the following functions:

train(model, device, train_loader, optimizer, epoch, train_acc, train_losses, runName, L1flag=False): Performs the training of the model using the backpropagation algorithm. It takes the model, device, training data loader, optimizer, epoch number, training accuracy list, training loss list, run name, and an optional L1 regularization flag as inputs. It returns the updated training accuracy and loss lists.

test(model, device, test_loader, test_acc, test_losses, runName): Evaluates the model using the test data. It takes the model, device, test data loader, test accuracy list, test loss list, and run name as inputs. It returns the updated test accuracy and loss lists, as well as the average test loss.

#### [visualize.py](https://github.com/adil22jaleel/era-v1-assignments/blob/main/s8_assignment/visualize.py)

The visualize.py script is used for visualizing the training process and results of a neural network model. It provides functions to plot various metrics such as input image , accuracy and loss during training.



### Analysing the Normalization outputs

1. Network with Group Normalization

- Train Accuracy: 78.58%
- Test Accuracy: 77.60%
- Misclassified Images
  
![groupnorm.png](https://github.com/adil22jaleel/era-v1-assignments/blob/main/s8_assignment/images/groupnorm.png)

2. Network with Layer Normalization

- Train Accuracy: 78.07%
- Test Accuracy: 76.45%
- Misclassified Images

![layernorm.png](https://github.com/adil22jaleel/era-v1-assignments/blob/main/s8_assignment/images/layernorm.png)

3. Network with Batch Normalization
- Train Accuracy: 78.18%
- Test Accuracy: 76.96%
- Misclassified Images
![batchnorm.png](https://github.com/adil22jaleel/era-v1-assignments/blob/main/s8_assignment/images/batchnorm.png)


### Accuracy & Loss Plots for Training and Test

![loss_plot](https://github.com/adil22jaleel/era-v1-assignments/blob/main/s8_assignment/images/loss_plot.png)

![accuracyplot.png](https://github.com/adil22jaleel/era-v1-assignments/blob/main/s8_assignment/images/accuracyplot.png)


### Final Analysis

In comparison, all three normalization techniques are having similar accuracy.

- For a specific image, group normalization conducts normalization for discrete channels. Increased group size improves the regularising impact.

- Since layer normalization normalizes every channel of every image, it performs the worst. This is a unique instance of group normalization involving only one group.

- Using batch normalization, all of the images in a batch are normalized for each channel.
