# CIFAR Image Classification Using Advanced Convolutions

## Objectives

1. Has the architecture to C1C2C3C40 (No MaxPooling, but 3 3x3 layers with stride of 2 instead) (If you can figure out how to use Dilated kernels here instead of MP or strided convolution)
2. Total RF must be more than 44
3. One of the layers must use Depthwise Separable Convolution
4. One of the layers must use Dilated Convolution
5. Use GAP (compulsory):- add FC after GAP to target #of classes (optional)
6. Use albumentation library and apply:
    - horizontal flip
    - shiftScaleRotate
    - coarseDropout (max_holes = 1, max_height=16px, max_width=1, min_holes = 1, min_height=16px, min_width=16px, fill_value=(mean of your dataset), mask_fill_value = None)
7. Achieve 85% accuracy, as many epochs as you want. Total Params to be less than 200k.

## Code Structure

### [dataset.py](https://github.com/adil22jaleel/era-v1-assignments/blob/main/s9_assignment/dataset.py)

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
    - HorizontalFlip randomly flips an image horizontally, i.e., mirroring it across the vertical axis. This can be used to help a model generalize to images that may be flipped in the real world.
    - ShiftScaleRotate randomly applies affine transforms to an image, including translation, scaling, and rotation. This can be used to help a model learn to recognize objects in different positions and sizes.
    - CoarseDropout randomly removes rectangular regions from an image. This can be used to help a model learn to recognize objects even when they are partially obscured.   

The following image shows the image post augmentation.
![image_augmentation](https://github.com/adil22jaleel/era-v1-assignments/blob/main/s9_assignment/images/albumentation.png)


### [model.py](https://github.com/adil22jaleel/era-v1-assignments/blob/main/s9_assignment/model.py)


#### Model Architecture
- Input Block
    -  convblock1: Applies a 3x3 convolution with 32 output channels, followed by ReLU activation and batch normalization.
- Convolution Block 1
    - convblock2: Consists of a series of operations:
        - 1x1 convolution with 160 output channels, followed by ReLU activation and batch normalization.
        - 3x3 depthwise separable convolution with 160 input and output channels, with a group size of 160. This is a depthwise separable convolution operation.
        - ReLU activation and batch normalization.
        - 1x1 convolution with 32 output channels, followed by ReLU activation, batch normalization, and dropout.
- Transition Block 1
    - convblock3: Applies a 3x3 convolution with 32 output channels and stride 2 for downsampling, followed by batch normalization.
- Convolution Block 2
    - convblock4: Similar to Convolution Block 1, but with different input and output channel sizes.
- Transition Block 2
    - convblock6: Applies a 3x3 convolution with 32 output channels and stride 2 for downsampling, followed by batch normalization.
- Convolution Block 3
    - convblock7: Similar to Convolution Block 1 and 2, but with different input and output channel sizes.
- Shortcut Connections
    - shortcut1: Applies a 1x1 convolution to the output of Convolution Block 2 for shortcut connections.
    - shortcut2: Applies a 1x1 convolution to the output of Convolution Block 3 for shortcut connections.
- Transition Block 3
    - convblock8: Applies a dilated 3x3 convolution with 64 output channels and a dilation factor of 2, followed by batch normalization.
- Convolution Block 4
    - convblock9: Similar to Convolution Block 1, 2, and 3, but with different input and output channel sizes.
- Output Block
    - gap: Applies average pooling with a kernel size of 4.
    - linear: Applies a linear transformation (fully connected layer) to the output of the average pooling layer to produce the final output logits.

The output logits are passed through a log softmax function for classification.


![model_archi](https://github.com/adil22jaleel/era-v1-assignments/blob/main/s9_assignment/images/model_summary_png.png)

#### Receptive Field Calculation

|Block |	Layer|	Input RF|	Input Size|	Jump In|	Stride|	Padding|	Kernel Size|	Dilation|	Eff. Kernel Size|	Output RF|	Output Size|	Jump Out|
|------|---------|----------|-------------|--------|----------|--------|---------------|------------|-------------------|------------|-------------|------------|
Input Block|	conv1|	1|	32|	1|	1|	1|	3|	1|	3|	3|	32|	1|
Conv Block 1|	conv2|	3|	32|	1|	1|	0|	1|	1|	1|	3|	32|	1|
Conv Block 1|	conv3|	3|	32|	1|	1|	1|	3|	1|	3|	5|	32|	1|
Conv Block 1|	conv4|	5|	32|	1|	1|	0|	1|	1|	1|	5|	32|	1|
Trans Block 1|	conv5|	5|	32|	1|	2|	1|	3|	1|	3|	7|	16|	2|
Conv Block 2|	conv6|	7|	16|	2|	1|	0|	1|	1|	1|	7|	16|	2|
Conv Block 2|	conv7|	7|	16|	2|	1|	1|	3|	1|	3|	11|	16|	2|
Conv Block 2|	conv8|	11|	16|	2|	1|	0|	1|	1|	1|	11|	16|	2|
Trans Block 2|	conv9|	11|	16|	2|	2|	1|	3|	1|	3|	15|	8|	4|
Conv Block 3|	conv10|	15|	8|	4|	1|	0|	1|	1|	1|	15|	8|	4|
Conv Block 3|	conv11|	15|	8|	4|	1|	1|	3|	1|	3|	23|	8|	4|
Conv Block 3|	conv12|	23|	8|	4|	1|	0|	1|	1|	1|	23|	8|	4|
Conv Block 3 + Skip|	conv13|	23|	8|	4|	1|	0|	1|	1|	1|	23|	8|	4|
Trans Block 2|	conv14|	23|	8|	4|	1|	0|	3|	2|	5|	39|	4|	4|
Conv Block 4|	conv15|	39|	4|	4|	1|	0|	1|	1|	1|	39|	4|	4|
Conv Block 4|	conv16|	39|	4|	4|	1|	1|	3|	1|	3|	47|	4|	4|
Conv Block 4|	conv17|	47|	4|	4|	1|	0|	1|	1|	1|	47|	4|	4|
Conv Block 4 + Skip|	conv18|	47|	4|	4|	1|	0|	1|	1|	1|	47|	4|	4|
Output Block|	gap|	47|	4|	4|	4|	0|	4|	1|	4|	59|	1|	16|


### [backpropgation.py](https://github.com/adil22jaleel/era-v1-assignments/blob/main/s9_assignment/backpropagation.py)


The backpropagation.py script implements the backpropagation algorithm for training neural networks. It contains functions for training and testing a model using the backpropagation algorithm.

The backpropagation.py script provides the following functions:

train(model, device, train_loader, optimizer, epoch, train_acc, train_losses, L1flag=False): Performs the training of the model using the backpropagation algorithm. It takes the model, device, training data loader, optimizer, epoch number, training accuracy list, training loss list, and an optional L1 regularization flag as inputs. It returns the updated training accuracy and loss lists.

test(model, device, test_loader, test_acc, test_losses): Evaluates the model using the test data. It takes the model, device, test data loader, test accuracy list, test loss list as inputs. It returns the updated test accuracy and loss lists, as well as the average test loss.

### [visualize.py](https://github.com/adil22jaleel/era-v1-assignments/blob/main/s9_assignment/visualize.py)

This script provides functions to visualize the misclassified images and plot the training and test statistics.

#### Functions

- plot_misclassified_images(model, test_loader, classes, device)
    - This function plots the misclassified images from the test dataset along with their actual and predicted labels.

- train_plot_stats(train_losses, train_accuracies)
    - This function plots the training loss and accuracy statistics.

- test_plot_stats(test_losses, test_accuracies)
    - This function plots the test loss and accuracy statistics.

### [utils.py](https://github.com/adil22jaleel/era-v1-assignments/blob/main/s9_assignment/utils.py)

This Python script defines a utility function for printing the summary of a PyTorch model using the torchsummary package. The function print_summary takes a model instance and an optional input_size parameter as input and displays a summary of the model's architecture.

## Inferences and Analysis

### Model Final Metrics

- Training accuracy: 78.82%
- Test accuracy: 85.28%
- Number of epochs model trained: 25
- Number of parameters: 178,178 parameters

### Training Log
```
EPOCH: 0
Loss=1.3156054019927979 Batch_id=390 Accuracy=43.81: 100%|██████████| 391/391 [00:20<00:00, 18.75it/s]

Test set: Average loss: 1.1188, Accuracy: 6038/10000 (60.38%)

EPOCH: 1
Loss=1.231472373008728 Batch_id=390 Accuracy=56.31: 100%|██████████| 391/391 [00:20<00:00, 19.35it/s]

Test set: Average loss: 0.9782, Accuracy: 6494/10000 (64.94%)

EPOCH: 2
Loss=0.9525799751281738 Batch_id=390 Accuracy=60.84: 100%|██████████| 391/391 [00:18<00:00, 20.64it/s]

Test set: Average loss: 0.8253, Accuracy: 7114/10000 (71.14%)

EPOCH: 3
Loss=1.070061445236206 Batch_id=390 Accuracy=64.05: 100%|██████████| 391/391 [00:18<00:00, 20.75it/s]

Test set: Average loss: 0.7924, Accuracy: 7258/10000 (72.58%)

EPOCH: 4
Loss=0.9134027361869812 Batch_id=390 Accuracy=66.21: 100%|██████████| 391/391 [00:20<00:00, 19.43it/s]

Test set: Average loss: 0.7345, Accuracy: 7479/10000 (74.79%)

EPOCH: 5
Loss=0.9380553364753723 Batch_id=390 Accuracy=68.12: 100%|██████████| 391/391 [00:18<00:00, 20.72it/s]

Test set: Average loss: 0.6604, Accuracy: 7716/10000 (77.16%)

EPOCH: 6
Loss=0.7564642429351807 Batch_id=390 Accuracy=69.51: 100%|██████████| 391/391 [00:18<00:00, 20.85it/s]

Test set: Average loss: 0.6255, Accuracy: 7814/10000 (78.14%)

EPOCH: 7
Loss=0.9048473238945007 Batch_id=390 Accuracy=70.81: 100%|██████████| 391/391 [00:20<00:00, 19.47it/s]

Test set: Average loss: 0.6243, Accuracy: 7852/10000 (78.52%)

EPOCH: 8
Loss=0.890572726726532 Batch_id=390 Accuracy=71.58: 100%|██████████| 391/391 [00:18<00:00, 20.76it/s]

Test set: Average loss: 0.5936, Accuracy: 7972/10000 (79.72%)

EPOCH: 9
Loss=1.0838863849639893 Batch_id=390 Accuracy=72.69: 100%|██████████| 391/391 [00:18<00:00, 20.80it/s]

Test set: Average loss: 0.5858, Accuracy: 7975/10000 (79.75%)

EPOCH: 10
Loss=0.6716275811195374 Batch_id=390 Accuracy=72.97: 100%|██████████| 391/391 [00:19<00:00, 19.71it/s]

Test set: Average loss: 0.5259, Accuracy: 8156/10000 (81.56%)

EPOCH: 11
Loss=0.7616040110588074 Batch_id=390 Accuracy=73.50: 100%|██████████| 391/391 [00:18<00:00, 20.83it/s]

Test set: Average loss: 0.5516, Accuracy: 8123/10000 (81.23%)

EPOCH: 12
Loss=0.7240789532661438 Batch_id=390 Accuracy=74.50: 100%|██████████| 391/391 [00:19<00:00, 19.87it/s]

Test set: Average loss: 0.5051, Accuracy: 8257/10000 (82.57%)

EPOCH: 13
Loss=0.6966513395309448 Batch_id=390 Accuracy=74.92: 100%|██████████| 391/391 [00:19<00:00, 19.72it/s]

Test set: Average loss: 0.5063, Accuracy: 8271/10000 (82.71%)

EPOCH: 14
Loss=0.6671422719955444 Batch_id=390 Accuracy=75.21: 100%|██████████| 391/391 [00:19<00:00, 20.51it/s]

Test set: Average loss: 0.5115, Accuracy: 8276/10000 (82.76%)

EPOCH: 15
Loss=0.7552340626716614 Batch_id=390 Accuracy=75.95: 100%|██████████| 391/391 [00:18<00:00, 20.96it/s]

Test set: Average loss: 0.4866, Accuracy: 8302/10000 (83.02%)

EPOCH: 16
Loss=0.46550053358078003 Batch_id=390 Accuracy=76.72: 100%|██████████| 391/391 [00:19<00:00, 19.66it/s]

Test set: Average loss: 0.4707, Accuracy: 8379/10000 (83.79%)

EPOCH: 17
Loss=0.744552731513977 Batch_id=390 Accuracy=76.61: 100%|██████████| 391/391 [00:18<00:00, 20.84it/s]

Test set: Average loss: 0.4784, Accuracy: 8366/10000 (83.66%)

EPOCH: 18
Loss=0.6969953179359436 Batch_id=390 Accuracy=76.89: 100%|██████████| 391/391 [00:19<00:00, 20.57it/s]

Test set: Average loss: 0.4590, Accuracy: 8448/10000 (84.48%)

EPOCH: 19
Loss=0.8309234380722046 Batch_id=390 Accuracy=77.66: 100%|██████████| 391/391 [00:20<00:00, 19.47it/s]

Test set: Average loss: 0.4706, Accuracy: 8410/10000 (84.10%)

EPOCH: 20
Loss=0.6827611327171326 Batch_id=390 Accuracy=77.49: 100%|██████████| 391/391 [00:19<00:00, 19.92it/s]

Test set: Average loss: 0.4591, Accuracy: 8404/10000 (84.04%)

EPOCH: 21
Loss=0.713283360004425 Batch_id=390 Accuracy=77.69: 100%|██████████| 391/391 [00:19<00:00, 20.55it/s]

Test set: Average loss: 0.4406, Accuracy: 8495/10000 (84.95%)

EPOCH: 22
Loss=0.596083402633667 Batch_id=390 Accuracy=78.21: 100%|██████████| 391/391 [00:20<00:00, 19.38it/s]

Test set: Average loss: 0.4500, Accuracy: 8450/10000 (84.50%)

EPOCH: 23
Loss=0.6029220819473267 Batch_id=390 Accuracy=78.38: 100%|██████████| 391/391 [00:21<00:00, 18.41it/s]

Test set: Average loss: 0.4332, Accuracy: 8483/10000 (84.83%)

EPOCH: 24
Loss=0.5489422082901001 Batch_id=390 Accuracy=78.82: 100%|██████████| 391/391 [00:19<00:00, 20.41it/s]

Test set: Average loss: 0.4353, Accuracy: 8528/10000 (85.28%)
```


### Accuracy and Loss Plots

![training_plots](https://github.com/adil22jaleel/era-v1-assignments/blob/main/s9_assignment/images/training_metrics.png)

![test_plots](https://github.com/adil22jaleel/era-v1-assignments/blob/main/s9_assignment/images/test_metrics.png)

### Misclassified Images
![misclassified_images](https://github.com/adil22jaleel/era-v1-assignments/blob/main/s9_assignment/images/misclassified.png)
