# Step by Step Design for digit classification on MNIST data using PyTorch

## Objective

The objective of this exercise is to create a powerful CNN model that can recognise handwritten numbers in the MNIST dataset. The neural network's design will be based on a few factors, including:

- 99.4% accuracy
- Less than or equal to 15 Epochs
- Less than 8000 Parameters
- Using modular code

## Model Design

We will go more into how we might design an architecture that can deliver the results we anticipate while sticking to the predetermined constraints.

## Step 1

### File Link: [Step 1 Notebook](https://pages.github.com/)

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

### File Link: [Step 2 Notebook](https://pages.github.com/)

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

### File Link: [Step 3 Notebook](https://pages.github.com/)

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