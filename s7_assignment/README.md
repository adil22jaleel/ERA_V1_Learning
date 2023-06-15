# Step by Step Design for digit classification on MNIST data using PyTorch

## Objective

The objective of this exercise is to create a powerful CNN model that can recognise handwritten numbers in the MNIST dataset. The neural network's design will be based on a few factors, including:

- 99.4% accuracy
- Less than or equal to 15 Epochs
- Less than 8000 Parameters
- Using modular code

## Model Design

We will go more into how we might design an architecture that can deliver the results we anticipate while sticking to the predetermined constraints.
There are a total of 3 notebooks that contains multiple codes of sequence of creating a CNN model. Apart from these notebooks, there are model.py and utils.py which have the following functions
- [model.py](https://github.com/adil22jaleel/era-v1-assignments/blob/s7/s7_assignment/model.py) : Contains various Neural Network Architectures that are used for the model development
- [utils.py](https://github.com/adil22jaleel/era-v1-assignments/blob/s7/s7_assignment/utils.py) : Contains the various data loading, training and testing codes for the model development
The notebook analysis is explained below.

## Step 1

### File Link: [Step 1 Notebook](https://github.com/adil22jaleel/era-v1-assignments/blob/s7/s7_assignment/S7_Step1.ipynb)

### Target:
1. Setting up a skeleton for the neural network
2. Setting a basic Architecture with GAP to remove the final layer

### Results:
- Parameters: 257,354   
- Best Train Accuracy: 99.37 %  (15th Epoch)   
- Best Test Accuracy: 99.15 %  (14th Epoch)   

### Analysis:
1. The skelton contain combination of multiple Convolution layers with 2 max pooling layer away from the output layer
2. The GAP layer is used to average the whole channel to a single value.
3. Max pooling is applied on two layers on different patches to maximise the content in that particular feature maps
4. The number of parameters are on higher side. This is because we are just create a sample skelton in the first step.
5. We have crossed the RF of 32 (which is more than the image size) with the skelton of the model

## Step 2

### File Link: [Step 2 Notebook](https://github.com/adil22jaleel/era-v1-assignments/blob/s7/s7_assignment/S7_Step2.ipynb)

### Target:
1. Introduce batch normalisation for performance improvement
2. Avoid any overfitting issues
3. Reduce the number of parameters 

### Results:
- Parameters: 7,216  
- Best Train Accuracy: 99.03 % (15th Epoch)   
- Best Test Accuracy: 99.28% (15th Epoch)   

### Analysis:

1. In the code 2 of this step, we are reducing the number of input and output channels to reduce the number of parameters and adding batch normalisation. The batch normalisation improves the performance of model training under each epoch

2. The code 2 model had overfitting issues still, with the difference between the training and testing accuracy being similar

3. To trim the overfitting issues, we introduced dropout with a dropout rate of 0.05 (This is added in the model.py) (code 3)

4. Now to reduce the number of parameters under 8k, 1x1 convolutions (code 4) were introduced along with additional convolution layer. The main purpose of a 1x1 convolution is to transform the channel dimension of the input feature maps. By altering the number of channels, it can change the dimensionality of the feature space.Now the number of parameters have reduced under 8k.

5. The training accuracy slowly in the last epoch crosses the 99% accuracy but test accuracy is not still hitting above 99.3%

6. Now we have attained all the possible approaches with the model building, now we can look to improve the accuracy by image augmentation and learning rate changes.

## Step 3

### File Link: [Step 3 Notebook](https://github.com/adil22jaleel/era-v1-assignments/blob/s7/s7_assignment/S7_Step3.ipynb)

### Target:
1. Acheive accuracy of 99.4 by involving image augmentation and if required by reduction of the learning rate 

2. Make sure the parameters are below 8k and 15 epochs

3. The model changes will be minimal and more will be on images and the learning optimizers.

### Results:
- Parameters: 7,216  
- Best Train Accuracy: 99.19 % (15th Epoch)   
- Best Test Accuracy: 99.45% (15th Epoch)   

### Analysis:
1. We are using the model 4 from the previous step to build model of accuracy above 99.4%

2. On the code 5, we are applying image augmentation by rotating the mnist image by angle of rotation between -7.0 and 7.0 degrees. The image augmentation transformation is only applied on the training dataset and not on test dataset

3. The model overfitting issues were completely sorted and we can see the gap between the training and testing accuracy was reducing, but the accuracy was not still hitting the 99.4 mark

4. To improve the model accuracy again, we introduced reducelronplateau as the learning rate scheduler method

5. ReduceLRonPlateau was set as patience as 2. This means that the LR will be same for the first 2 epochs with no improvement, and will only decrease the LR after the 3rd epoch if the loss still hasn’t improved 

6. For ReduceLRonPlateau, we are calling separate function for the testing, because for the learning rate change we have to pass the test loss. The function is specified in the utils.py as test_model_plateau

7. It is visible that the training set slowly hits the 99% accuracy in the last few epochs whilst the testing accuracy crosses the 99.4 mark because of the learning rate change. 

The following is the log for the last model

'''
EPOCH: 1

  0%|          | 0/469 [00:00<?, ?it/s]/content/model.py:155: UserWarning: Implicit dimension choice for log_softmax has been deprecated. Change the call to include dim=X as an argument.
  return F.log_softmax(x)
Loss=0.15595975518226624 Batch_id=468 Accuracy=90.43: 100%|██████████| 469/469 [00:29<00:00, 16.15it/s]

Test set: Average loss: 0.0795, Accuracy: 9773/10000 (97.73%)

EPOCH: 2

Loss=0.040991537272930145 Batch_id=468 Accuracy=97.40: 100%|██████████| 469/469 [00:20<00:00, 22.61it/s]

Test set: Average loss: 0.0435, Accuracy: 9871/10000 (98.71%)

EPOCH: 3

Loss=0.0806039571762085 Batch_id=468 Accuracy=97.98: 100%|██████████| 469/469 [00:21<00:00, 21.51it/s]

Test set: Average loss: 0.0357, Accuracy: 9895/10000 (98.95%)

EPOCH: 4

Loss=0.023201346397399902 Batch_id=468 Accuracy=98.19: 100%|██████████| 469/469 [00:21<00:00, 21.90it/s]

Test set: Average loss: 0.0355, Accuracy: 9886/10000 (98.86%)

EPOCH: 5

Loss=0.012452361173927784 Batch_id=468 Accuracy=98.40: 100%|██████████| 469/469 [00:22<00:00, 21.16it/s]

Test set: Average loss: 0.0294, Accuracy: 9911/10000 (99.11%)

EPOCH: 6

Loss=0.03821573406457901 Batch_id=468 Accuracy=98.47: 100%|██████████| 469/469 [00:21<00:00, 21.41it/s]

Test set: Average loss: 0.0281, Accuracy: 9914/10000 (99.14%)

EPOCH: 7

Loss=0.01874718628823757 Batch_id=468 Accuracy=98.64: 100%|██████████| 469/469 [00:22<00:00, 21.20it/s]

Test set: Average loss: 0.0236, Accuracy: 9933/10000 (99.33%)

EPOCH: 8

Loss=0.027351198717951775 Batch_id=468 Accuracy=98.68: 100%|██████████| 469/469 [00:20<00:00, 22.69it/s]

Test set: Average loss: 0.0268, Accuracy: 9916/10000 (99.16%)

EPOCH: 9

Loss=0.03188159689307213 Batch_id=468 Accuracy=98.73: 100%|██████████| 469/469 [00:20<00:00, 22.44it/s]

Test set: Average loss: 0.0219, Accuracy: 9935/10000 (99.35%)

EPOCH: 10

Loss=0.02450401335954666 Batch_id=468 Accuracy=98.78: 100%|██████████| 469/469 [00:22<00:00, 20.99it/s]

Test set: Average loss: 0.0229, Accuracy: 9922/10000 (99.22%)

EPOCH: 11

Loss=0.02112734317779541 Batch_id=468 Accuracy=98.81: 100%|██████████| 469/469 [00:22<00:00, 21.11it/s]

Test set: Average loss: 0.0254, Accuracy: 9921/10000 (99.21%)

EPOCH: 12

Loss=0.03854614868760109 Batch_id=468 Accuracy=98.87: 100%|██████████| 469/469 [00:21<00:00, 21.43it/s]

Test set: Average loss: 0.0246, Accuracy: 9922/10000 (99.22%)

EPOCH: 13

Loss=0.006861161440610886 Batch_id=468 Accuracy=99.13: 100%|██████████| 469/469 [00:21<00:00, 21.50it/s]

Test set: Average loss: 0.0193, Accuracy: 9942/10000 (99.42%)

EPOCH: 14

Loss=0.007884729653596878 Batch_id=468 Accuracy=99.13: 100%|██████████| 469/469 [00:20<00:00, 22.46it/s]

Test set: Average loss: 0.0185, Accuracy: 9944/10000 (99.44%)

EPOCH: 15

Loss=0.04473526403307915 Batch_id=468 Accuracy=99.19: 100%|██████████| 469/469 [00:20<00:00, 22.43it/s]

Test set: Average loss: 0.0186, Accuracy: 9945/10000 (99.45%)
'''
