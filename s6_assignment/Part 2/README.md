# MNIST Digit Classification using Neural Networks

In this exercise, we will be using the MNIST data for classifying handwritten digits using a convolutional layer. We will stick to the following operational parameters as well to see how a CNN can be efficiently configured - 
- 99.4% validation accuracy
- Less than 20k Parameters
- Less than 20 Epochs
- Use Batch Normalization and  Dropout,
- A Fully connected layer and have used GAP (Optional). 

## Model Architecture
```python
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.block0 = nn.Sequential(
          nn.Conv2d(1, 16, 3, bias=False),
          nn.ReLU(),
          nn.BatchNorm2d(16),
          nn.Dropout(0.1),
        )
        self.block1 = nn.Sequential(
          nn.Conv2d(16, 16, 3, bias=False),
          nn.ReLU(),
          nn.BatchNorm2d(16),
          nn.Dropout(0.1)
        )
        self.block1d = nn.Sequential(
          nn.Conv2d(16, 16, 3, bias=False),
          nn.ReLU(),
          nn.BatchNorm2d(16),
          nn.Dropout(0.1),
        )
        self.block2 = nn.Sequential(
          nn.Conv2d(16, 32, 3, bias=False),
          nn.ReLU(),
          nn.BatchNorm2d(32),
          nn.Dropout(0.1),
        )
        self.block3 = nn.Sequential(
          nn.Conv2d(32, 16, 3, bias=False),
          nn.ReLU(),
          nn.BatchNorm2d(16),
        )
        self.block4 = nn.Sequential(
          nn.Conv2d(16, 10, 1, bias=False),
        )
        self.pool1 = nn.AvgPool2d(2, 2)
        self.pool2 = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.pool1(self.block1(self.block1d(self.block0(x))))
        x = self.block4(self.pool2(self.block3(self.block2(x))))
        x = x.view(-1, 10)
        return F.log_softmax(x)

```

-  Input: The model takes grayscale images as input, with a single channel.

**Block 0 of Model**
- The first convolutional layer (conv1) performs a 2D convolution on the input, using a kernel size of 3x3 and generating 16 output channels.
- We are applying ReLU as activation function.ReLU is an activation function that introduces the property of non-linearity to a deep learning model and solves the vanishing gradients issue
- Next is the batch normalisation. Batch normalization is a method used to make training of artificial neural networks faster and more stable through normalization of the layers' inputs by re-centering and re-scaling.Batch normalization (bn1) is applied to the output of conv1.
- The output of BN passed through a 2D dropout layer and then through the ReLU activation function.The Dropout layer randomly sets input units to 0 with a frequency of rate at each step during training time, which helps prevent overfitting.
- We have bias set as False for the entire approach

**Block 1 of Model**
- The output of block 0 is passed through the second convolutional layer (conv2), which has a kernel size of 3x3 and produces 16 output channels.
- Batch normalization is applied to the output of conv2.
- The output is then passed through a 2D dropout layer (drop) and ReLU activation (F.relu).

**Block 1d of Model**
- It is similar to the block 1 of the model

**Block 2 of Model**

- The output of block 1d is passed through the second convolutional layer (conv2), which has a kernel size of 3x3 and produces 32 output channels.
- Batch normalization is applied to the output of conv2.
- The output is then passed through a 2D dropout layer (drop) and ReLU activation (F.relu).

**Block 3 of Model**

- The output of block 2 is passed through the second convolutional layer (conv2), which has a kernel size of 3x3 and produces 32 output channels.
- Batch normalization is applied to the output of conv2.
- The output is then passed to ReLU activation (F.relu).

**Block 4 of Model**

- The output of block3 is passed through one  convolutional layer with 10 output channels


**Line 1 of Forward Function**
- In the line 1 of forward function, we are applying average pooling after the block0 + block1 + block1d

**Line 2 of Forward Function**
- - In the line 2 of forward function, we are applying adaptive average pooling after the block2 + block3, this will globally average the feature maps.This is called GAP layer.

**Line 3 of Forward Function**
- x.view(-1, 10) The view method returns a tensor with the same data as the self tensor (which means that the returned tensor has the same number of elements), but with a different shape. The softmax function (F.log_softmax) is applied to obtain the final output probabilities for each class.

## Model Summary

```        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 26, 26]             144
              ReLU-2           [-1, 16, 26, 26]               0
       BatchNorm2d-3           [-1, 16, 26, 26]              32
           Dropout-4           [-1, 16, 26, 26]               0
            Conv2d-5           [-1, 16, 24, 24]           2,304
              ReLU-6           [-1, 16, 24, 24]               0
       BatchNorm2d-7           [-1, 16, 24, 24]              32
           Dropout-8           [-1, 16, 24, 24]               0
            Conv2d-9           [-1, 16, 22, 22]           2,304
             ReLU-10           [-1, 16, 22, 22]               0
      BatchNorm2d-11           [-1, 16, 22, 22]              32
          Dropout-12           [-1, 16, 22, 22]               0
        AvgPool2d-13           [-1, 16, 11, 11]               0
           Conv2d-14             [-1, 32, 9, 9]           4,608
             ReLU-15             [-1, 32, 9, 9]               0
      BatchNorm2d-16             [-1, 32, 9, 9]              64
          Dropout-17             [-1, 32, 9, 9]               0
           Conv2d-18             [-1, 16, 7, 7]           4,608
             ReLU-19             [-1, 16, 7, 7]               0
      BatchNorm2d-20             [-1, 16, 7, 7]              32
AdaptiveAvgPool2d-21             [-1, 16, 1, 1]               0
           Conv2d-22             [-1, 10, 1, 1]             160
================================================================
Total params: 14,320
Trainable params: 14,320
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.96
Params size (MB): 0.05
Estimated Total Size (MB): 1.02
```

## Model Usage
1. Define the train and test dataloaders.

    ```python
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
        batch_size=batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
        batch_size=batch_size, shuffle=True, **kwargs)
    ```

- Train Loader:

    - The datasets.MNIST function is used to create a dataset object for the MNIST training set.
    - The data can be downloaded if neceassary and will be available on the ./data folder
    - A transforms.Compose object is used to create a series of transformation
    - The transformations include converting the images to tensors (transforms.ToTensor()) and normalizing the pixel values with mean 0.1307 and standard deviation 0.3081 (transforms.Normalize((0.1307,), (0.3081,))).
    - The batch_size, shuffle, and other parameters (collected in the **kwargs variable) are specified to control the behavior of the data loader.

- Test Loader:
    - Similar to the train loader, this will be used for the test data
    - The transforms are same by Compose function

2. Definition of Train and Test
- train()

    ```python
    from tqdm import tqdm
    def train(model, device, train_loader, optimizer, epoch):
        model.train()
        pbar = tqdm(train_loader)
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            pbar.set_description(desc= f'loss={loss.item()} batch_id={batch_idx}')
    ```
    - tqdm library, provides a progress bar to track the training progress.
    - The train function is defined, taking the following parameters:
        - model: The model to be trained.
        - device: The device (CPU or GPU) on which the training will be performed.
        - train_loader: The data loader providing the training samples.
        - optimizer: The optimizer used for updating the model's parameters.
        - epoch: The current epoch number.
    - The function enters a loop that iterates over the mini-batches of training data using enumerate(train_loader).
    - For each batch, the input data and corresponding target labels are retrieved (data, target = data.to(device), target.to(device)), and they are moved to the specified device if available.
    - The optimizer's gradients are cleared using optimizer.zero_grad(). This is for the backward pass.
    - The input data is passed through the model to obtain the output predictions (output = model(data)).
    - The negative log-likelihood loss (F.nll_loss) is computed by comparing the model's output with the target labels (loss = F.nll_loss(output, target)).
    - The gradients are computed by performing backpropagation through the network (loss.backward()).
    - The optimizer updates the model's parameters based on the computed gradients (optimizer.step()).

- test()
```python
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  
            pred = output.argmax(dim=1, keepdim=True)  
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
```
- The functions takes the following as parameters-
    - model: The trained model to be evaluated.
    - device: The device (CPU or GPU) on which the evaluation will be performed.
    - test_loader: The data loader providing the test samples.
    Set the model to evaluation mode: model.eval()
- The variables for tracking the test loss and the number of correct predictions: test_loss = 0 and correct = 0
- torch.no_grad() helps in disabling the gradient calculation and backpropagation during testing:
- Move the input data and target labels are moved the device, here is it specific to CUDA device
- Forward pass: In this function we are passing the input data to model
- Compute the loss: Calculate the negative log-likelihood loss (nll_loss)and sum it up: test_loss += F.nll_loss(output, target, reduction='sum').item()
- Calculate the number of correct predictions-
    - Get the predicted class labels by selecting the index of the highest log-probability: pred = output.argmax(dim=1, keepdim=True)
    - Compare the predicted labels with the target labels and count the number of correct predictions: correct += pred.eq(target.view_as(pred)).sum().item()
- Calculate the average test loss: test_loss /= len(test_loader.dataset)
- Display the average loss, the number of correct predictions, the total number of test samples, and the accuracy percentage using print statement. The precision of accuracy can be changed on the floating point number.

By calculating the average loss and accuracy, this function assesses the model's performance on the test dataset. The accuracy is calculated by iteratively going over the test samples, calculating the loss, and comparing the predicted labels to the actual labels. To give a summary of the model's performance on the test set, the results are then printed.

3. Running the model
```python
model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

for epoch in range(1, 20):
    print("Epoch number ",epoch)
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)
```
- Model and Optimizer Initialization:
    - The Net model is instantiated and sent to the specified device: model = Net().to(device).
    - The optimizer (Stochastic Gradient Descent - SGD) is created with a learning rate of 0.01 and momentum of 0.9
- Training and Testing Loop:
    - A loop is executed for each epoch from 0 to 19 (the epoch number)
    - The train function is called, passing the model, device, training data loader, optimizer, and the current epoch number as arguments: train(model, device, train_loader, optimizer, epoch).
    - The test function is called, passing the model, device, and test data loader as arguments: test(model, device, test_loader).
    - The testing function evaluates the model's performance on the test set and prints the average loss and accuracy.
- This code runs testing after each of the model's 19 training iterations. It provides visibility into the training progress by printing the epoch number. Every epoch, the training and testing functions are used to update the model's parameters and assess how well it performs on the test set.


## Training Loss

The following shows how the test accuracy looks for each epoch

```Epoch number  1
  0%|          | 0/469 [00:00<?, ?it/s]<ipython-input-2-3f84d89a3834>:44: UserWarning: Implicit dimension choice for log_softmax has been deprecated. Change the call to include dim=X as an argument.
  return F.log_softmax(x)
loss=0.08337432146072388 batch_id=468: 100%|██████████| 469/469 [00:21<00:00, 21.39it/s]

Test set: Average loss: 0.0756, Accuracy: 9823/10000 (98.23%)

Epoch number  2
loss=0.036148156970739365 batch_id=468: 100%|██████████| 469/469 [00:21<00:00, 21.42it/s]

Test set: Average loss: 0.0482, Accuracy: 9869/10000 (98.69%)

Epoch number  3
loss=0.061487842351198196 batch_id=468: 100%|██████████| 469/469 [00:22<00:00, 20.97it/s]

Test set: Average loss: 0.0369, Accuracy: 9904/10000 (99.04%)

Epoch number  4
loss=0.03713545948266983 batch_id=468: 100%|██████████| 469/469 [00:21<00:00, 21.36it/s]

Test set: Average loss: 0.0382, Accuracy: 9888/10000 (98.88%)

Epoch number  5
loss=0.050500694662332535 batch_id=468: 100%|██████████| 469/469 [00:23<00:00, 20.32it/s]

Test set: Average loss: 0.0289, Accuracy: 9920/10000 (99.20%)

Epoch number  6
loss=0.03411164507269859 batch_id=468: 100%|██████████| 469/469 [00:22<00:00, 21.17it/s]

Test set: Average loss: 0.0278, Accuracy: 9920/10000 (99.20%)

Epoch number  7
loss=0.007832867093384266 batch_id=468: 100%|██████████| 469/469 [00:22<00:00, 21.30it/s]

Test set: Average loss: 0.0261, Accuracy: 9925/10000 (99.25%)

Epoch number  8
loss=0.024782178923487663 batch_id=468: 100%|██████████| 469/469 [00:21<00:00, 21.95it/s]

Test set: Average loss: 0.0234, Accuracy: 9928/10000 (99.28%)

Epoch number  9
loss=0.02867930568754673 batch_id=468: 100%|██████████| 469/469 [00:21<00:00, 22.21it/s]

Test set: Average loss: 0.0224, Accuracy: 9931/10000 (99.31%)

Epoch number  10
loss=0.01854584366083145 batch_id=468: 100%|██████████| 469/469 [00:21<00:00, 21.99it/s]

Test set: Average loss: 0.0194, Accuracy: 9934/10000 (99.34%)

Epoch number  11
loss=0.035152558237314224 batch_id=468: 100%|██████████| 469/469 [00:21<00:00, 22.11it/s]

Test set: Average loss: 0.0236, Accuracy: 9931/10000 (99.31%)

Epoch number  12
loss=0.005101292859762907 batch_id=468: 100%|██████████| 469/469 [00:21<00:00, 22.19it/s]

Test set: Average loss: 0.0193, Accuracy: 9936/10000 (99.36%)

Epoch number  13
loss=0.08994728326797485 batch_id=468: 100%|██████████| 469/469 [00:21<00:00, 21.83it/s]

Test set: Average loss: 0.0191, Accuracy: 9939/10000 (99.39%)

Epoch number  14
loss=0.008849493227899075 batch_id=468: 100%|██████████| 469/469 [00:21<00:00, 21.83it/s]

Test set: Average loss: 0.0190, Accuracy: 9936/10000 (99.36%)

Epoch number  15
loss=0.05290992930531502 batch_id=468: 100%|██████████| 469/469 [00:21<00:00, 21.95it/s]

Test set: Average loss: 0.0171, Accuracy: 9945/10000 (99.45%)

Epoch number  16
loss=0.0466175377368927 batch_id=468: 100%|██████████| 469/469 [00:21<00:00, 21.62it/s]

Test set: Average loss: 0.0186, Accuracy: 9938/10000 (99.38%)

Epoch number  17
loss=0.01837375946342945 batch_id=468: 100%|██████████| 469/469 [00:21<00:00, 21.66it/s]

Test set: Average loss: 0.0208, Accuracy: 9931/10000 (99.31%)

Epoch number  18
loss=0.005579816177487373 batch_id=468: 100%|██████████| 469/469 [00:21<00:00, 21.60it/s]

Test set: Average loss: 0.0174, Accuracy: 9943/10000 (99.43%)

Epoch number  19
loss=0.011081268079578876 batch_id=468: 100%|██████████| 469/469 [00:21<00:00, 21.69it/s]

Test set: Average loss: 0.0172, Accuracy: 9943/10000 (99.43%)

'''