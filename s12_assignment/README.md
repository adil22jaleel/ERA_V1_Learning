# CIFAR-10 Classification using Lightning Module
This repository contains code for training a ResNet-based model for CIFAR-10 classification using PyTorch Lightning. The model architecture includes custom ResBlocks for feature extraction. Albumentations library is used for data augmentation. The code also includes functionality for GradCAM visualization of misclassified images.
## Hugging Face Space Link- [Hugging Face Space](https://huggingface.co/spaces/adil22jaleel/CIFAR_ResnetModel)
## Understanding the different functions

### ResBlock

The ResBlock is a custom PyTorch module designed to create a residual block, which is a fundamental building block in the ResNet architecture. The ResNet architecture is widely used for deep learning tasks, particularly for image classification, due to its ability to effectively train very deep neural networks.

#### Forward Method

The forward method defines the computation flow within the ResBlock. It takes the input tensor x and applies a sequence of operations:

A 2D convolutional layer with batch normalization and ReLU activation, transforming the input tensor with in_ch channels to out_ch channels.
Another 2D convolutional layer with batch normalization and ReLU activation, further processing the output from the previous step.
The output of the block is the final output tensor obtained after applying the above operations to the input tensor.

### CifarDatasetWAlbumentations

The CifarDatasetWAlbumentations class is a custom dataset class that extends the PyTorch Dataset class. It is used to load and preprocess data for the CIFAR-10 dataset with the help of the Albumentations library. Albumentations provides various image augmentations, allowing for improved model generalization and performance.

## Lightning Module- LitCIFAR_ResNet

The LitCIFAR_ResNet class is a PyTorch Lightning Module for training and evaluating a ResNet-based model for CIFAR-10 classification. This class extends the LightningModule provided by PyTorch Lightning, which simplifies the training process by abstracting away many boilerplate code and providing useful features like distributed training, logging, and checkpointing.

The constructor of the LitCIFAR_ResNet class takes three optional parameters:

- data_dir: The path to the directory where the CIFAR-10 dataset is stored. If not provided, it defaults to "./Data".
- BATCH_SIZE: The batch size to be used during training and evaluation. If not provided, it defaults to 512.
- learning_rate: The initial learning rate for the optimizer. If not provided, it defaults to 1e-7.

### Model Architecture
The model architecture is based on the ResNet design, which includes custom ResBlocks for feature extraction. The model is composed of the following components:

- preplayer: The initial 2D convolutional layer followed by batch normalization and ReLU activation.
- layer1_X: The first part of the first residual block, which includes a 2D convolutional layer, max-pooling, batch normalization, and ReLU activation.
- layer1_R: The second part of the first residual block, which is an instance of the custom ResBlock class.
- layer2_X: The first part of the second residual block, which includes a 2D convolutional layer, max-pooling, batch normalization, and ReLU activation.
- layer3_X: The first part of the third residual block, which includes a 2D convolutional layer, max-pooling, batch normalization, and ReLU activation.
- layer3_R: The second part of the third residual block, which is another instance of the custom ResBlock class.
- maxpool_4: A max-pooling layer with a kernel size of 4.
- linear: The final fully connected layer that produces the output logits.

#### Forward Method
The forward method defines the forward pass of the model. It takes an input tensor x and passes it through the various layers of the model to produce the output logits. The output logits are then transformed using the log-softmax function to obtain the final probability distribution over the classes.

### Dataset Transformations
The class defines two sets of transformations for the CIFAR-10 dataset:

- train_transforms: A set of transformations to be applied to the training data. It includes normalization, padding, random cropping, horizontal flipping, cutout augmentation, and converting the data to tensors.
- val_transforms: A set of transformations to be applied to the validation and test data. It includes normalization and converting the data to tensors.

### Accuracy Metric
The model uses the Accuracy metric from the torchmetrics library to track accuracy during training and validation. It is a multiclass accuracy metric with 10 classes corresponding to the CIFAR-10 categories.

### Training, Validation, and Testing Steps
The class defines the training_step, validation_step, and test_step methods to perform the forward pass, calculate the loss, and update the accuracy metric during training, validation, and testing, respectively.

### Optimizer and Learning Rate Scheduler
The model uses the Adam optimizer with weight decay for parameter optimization. It also utilizes the OneCycleLR learning rate scheduler to automatically adjust the learning rate during training.

### Data Loading
The class provides methods for preparing and setting up the CIFAR-10 dataset for training, validation, and testing. It also includes a method, collect_misclassified_images, for collecting misclassified images during testing for further analysis.

### Visualization
The class includes methods, show_misclassified_images and get_gradcam_images, for visualizing misclassified images and GradCAM visualizations of misclassified images, respectively.

## Inferences

TensorBoard is a visualization tool provided by TensorFlow that allows you to monitor and analyze various aspects of your machine learning models during training and evaluation. The following are the plots from the tensorboard.

### Plots 
Train Accuracy![train_acc](https://github.com/adil22jaleel/era-v1-assignments/blob/main/s12_assignment/images/train_acc.jpg)
Train Loss![train_loss](https://github.com/adil22jaleel/era-v1-assignments/blob/main/s12_assignment/images/train_loss.jpg)
Validation Accuracy![val_acc](https://github.com/adil22jaleel/era-v1-assignments/blob/main/s12_assignment/images/val_acc.jpg)
Validation Loss![val_loss](https://github.com/adil22jaleel/era-v1-assignments/blob/main/s12_assignment/images/val_loss.jpg)

Model Test Accuracy: 90%
Model Test Loss: 0.30
