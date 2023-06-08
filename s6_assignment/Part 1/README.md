# Understanding Backpropagation

An technique called backpropagation, also known as backward propagation of mistakes, is made to check for errors as data travels backward from input nodes to output nodes. With regard to weight values for the various inputs, artificial neural networks compute a gradient descent using backpropagation as their learning algorithm. In order to minimise the discrepancy between the two, connection weights are adjusted in the systems by comparing desired outputs to achieved system outputs.The name back propagation comes because of the weight modification in reverse, from output to input.

## 1. Problem Statement

### 1.1 Background

Backpropagation is a fundamental algorithm in the field of artificial neural networks. The technique backpropagation, also known as backward propagation of mistakes, is made to check for errors as data travels backward from input nodes to output nodes. With regard to weight values for the various inputs, artificial neural networks compute a gradient descent using backpropagation as their learning algorithm.Understanding the backpropagation steps is crucial for grasping the inner workings of neural networks and training them effectively. 

### 1.2 Objective:
The objective of this project is to create a GitHub README file that provides a detailed explanation of the backpropagation steps. The README should be beginner-friendly, enabling individuals with minimal background in neural networks to understand and implement backpropagation.

### 1.3 Requirements:
To accomplish this objective, the following requirements are performed:

* Step-by-step Explanation: The README file should break down the backpropagation algorithm into clear and sequential steps. Each step should be explained in a concise yet comprehensive manner, highlighting its purpose and how it contributes to the overall training process.
 

## Introduction to Backpropgation (BP)

Backpropagation is a fundamental algorithm in the field of artificial neural networks, playing a vital role in training these networks to perform complex tasks. It serves as a powerful tool for adjusting the weights and biases of the neural network, allowing it to learn and make accurate predictions.

It involves two main phases: 
* the forward pass
* the backward pass. 

1. Forward Pass: The network takes an input and performs a series of computations, propagating the signal through the layers to produce an output.

2. Backward Pass: The error between the predicted output and the actual output is calculated, and this error is then backpropagated through the layers, updating the weights and biases along the way.

In this project, we will delve into the details of backpropagation, explaining its steps, concepts, and implementation. Through a step-by-step breakdown, illustrative examples, we will break the backpropagation algorithm and equip you with the knowledge to implement and train neural networks effectively.

## Visualizing the BP
![backpropagation_image](./Images/BP_main.jpg?raw=true)

The above diagram shows the simple flow of an artificial neural network with
* one input layer (i1 and i2)
* one hidden layer (h1 and h2)
* one output layer (o1 and o2)

Apart from the base layers there are other features like the weights, activation function and errors calculated which is important to know for the calculation of backpropagation algorithm

* The connections between two nodes across layers are called weights and default values are set between the weights. (W1=0.15, where W1 is the weight between input neuron 1 and hidden neuron 1)
* Similarily to ease the explaination, W8=0.55 is the weight between hidden neuron 2 and output neuron 2
* Usually sigma is the activation function applied over any function, the node a_h1 is output of activation function applied on h1
* The last node in red, is the total error calculated which initiates the backward pass
* t1 and t2 represents the target value
* E is the total loss that is calculated

## The Math behind BP

Now we have seen the different elements that are important for backpropagation algorithm to work, lets drill down through the algorithm and explain the various math that is used for calculating the outcomes. The backpropagation algorithm will go forward and backward on N number of iterations based on how the loss goes, the following step by step process helps in calculating for one complete iteration of backprop. Initially consider the following values

$$i1= 0.05 , i2=0.1$$
$$w1= 0.15, w2=0.2 , w3=0.25 , w4=0.3$$
$$w5= 0.4, w6=0.45 , w7=0.50 , w8=0.55$$

### 1. Calculating the Hidden Layer and its Activation Function

**Objective: To calculate h1, h2 , a_h1, a_h2**

The hidden layer is calculated based on the input layer and the initialised weights. 

**Hidden Layer formula**
$$h1=w1*i1 + w2*i2$$
$$h2=w3*i1 + w4*i2$$
$$h1=0.0275, h2=0.0425$$

Now we know the values for h1 and h2, we can calculate the activation function over it. In this example we are using the sigmoid function.

**Sigmoid Function**
$$σ(x) = 1/(1 + exp(-x))$$

Hence the activation function a_h1 and a_h2 is calculated as
$$a\\_h1=1/(1 + exp(-h1))=0.506874567$$
$$a\\_h2=1/(1 + exp(-h2))=0.510623401$$


### 2. Calculating the Output Layer and its Activation Function

**Objective: To calculate o1, o2 , a_o1, a_o2**

If the hidden layer is calculated between inputs layer and input weights, the output layer is calculated based on the hidden layer and hidden weights. 

From step 1, we know the values for h1 ,h2 ,a_h1 and a_h2. We can use the similar for calculation.

**Output Layer formula**
$$o1=w5*a_h1 + w6*a\\_h2$$
$$o2=w7*a_h1 + w8*a\\_h2$$

After applying the sigmoid activation function
$$a\\_o1=0.606477732$$
$$a\\_o2=0.630480835 $$

### 3. Calculating the Error

**Objective: To calculate E**

To calculate the total error, we need to calculate the error that is obtained from all the output nodes in the output layer.

**Error calculation formula**
$$ E1=½ * (t1 - a\_o1)²$$
$$ E2=½ * (t2- a\_o2)² $$
where t1 and t2 are the target variable names. We have assumed the target here as t1=t2=0.5

The total error is the sum of all the errors from the output layer. Total Error is
$$ E= E1 + E2$$
and is calculated as 
$$ E= E1 +E2=0.005668754 + 0.008512624$$
$$ Total Error E = 0.014181378 $$

### 4. Calculating the Gradients for Hidden Layer Weights

How we have finished our first iteration of the feed forward network, we will progress with the backward network (from output towards input) For any backpropagation problem, the most important factor is the gradient descent. We need to optimize weight to minimize error. To do this we need to find the derivative of the Error with respect to the weight. This derivative is called Gradient.

Gradient is calculated for each weight in the network. 

Now we can familiarise the gradient calculation for the weights near to output first. 
For gradient calculation for the weights w5-w8, we are dividing the code into 3 blocks

#### 4.1 Gradient calculation Block 1 (w5-w8) 

Gradient for w5 is expressed as
$$∂E\_total/∂w5 = ∂(E1 + E2)/∂w5$$
$$∂E\_total/∂w5 = ∂E1/∂w5$$
$$∂E\_total/∂w5 = `∂E1/∂a_o1*∂a_o1/∂o1*∂o1/∂w5`$$

In the above second equation, the E2 component becomes zero because there weights w5 does not have derivaties or dependencies on the second error output provided

The third line explains the chain rule. We are expressing the partial derivative of w5 in terms of a_o1 and o1

#### 4.2 Gradient calculation Block 2 (w5-w8) 

From the above block, we have to find respective values for each of the value that is multipled in the chain product

$$∂E1/∂a_o1 =  ∂(½ * (t1 - a\_o1)²)/∂a_o1 = (a\_o1 - t1)$$
*Partial derivative of activation function of o1 to the first error loss*
$$∂a\_o1/∂o1 =  ∂(σ(o1))/∂o1 = a\_o1*(1-a\_o1)$$
*Partial derivative of o1 to the activation function of o1*
$$∂o1/∂w5 = a\_h1$$
*Partial derivative of weight5 on top of the output neuron 1*

Multiplying all together, we get 
$$∂E\_total/∂w5 = (a\_o1 - t1) * a\_o1 * (1 - a\_o1) *  a\_h1$$

Similarily, we can calculate for the gradient for w6,w7,w8
$$∂E\_total/∂w6 = (a\_o1 - t1) * a_o1 * (1 - a\_o1) *  a_h2$$
$$∂E\_total/∂w7 = (a\_o2 - t2) * a_o2 * (1 - a\_o2) *  a_h1$$
$$∂E\_total/∂w8 = (a\_o2 - t2) * a_o2 * (1 - a\_o2) *  a_h2$$


By the calculations from above, we know
$$ a\_o1=0.606477732 , a\_o2=0.630480835 $$ 
$$ t1=0.5 , t2=0.5 $$
$$a\_h1=0.506874567 , a\_h2=0.510623401$$

*hence*

$$ ∂E/∂w5=0.012880819 ,∂E/∂w6=0.012976085 $$
$$ ∂E/∂w7=0.015408348, ∂E/∂8=0.015522308 $$

This is the gradient of the weight calculated for weights w5,w6,w7,w8
		
### 5. Calculating the Gradients for Hidden Layer Weights

Next up we need to find the gradients for the weights of the input layers. 

Let us consider the case of weight w1. For the gradients of we need to first find the gradients of a_h1 with respect to the error.

The gradient of the total error with respect to a_h1 can be denoted as 
$∂E_total/∂a\_h1 = ∂E_1/∂a\_h1 + ∂E_1/∂a\_h2$ 

$$∂E1/∂a\_h1 = (a_01 - t1) * a_o1 * (1 - a_o1) * w5$$
$$∂E2/∂a\_h1 = (a_02 - t2) * a_o2 * (1 - a_o2) * w7$$
Similar to the gradient descent of the hidden layer, we can derive $∂E_total/∂a\_h1 = ∂E_1/∂a\_h1 = ∂E_1/∂a\_o1 * ∂a\_o1/∂o1 * ∂o1/∂a\_h1$				

The equations can be written as 
$$∂E_1/∂a\_o1 =  ∂(½  *  (t1 - a\_o1)²)/∂a\_o1 = (a\_o1 - t1)$$
$$∂a\_o1/∂o1 =  ∂(σ(o1))/∂o1 = a\_o1  *  (1 - a\_o1)$$
$$∂o1/∂a\_h1 = w_5$$

By bring all the equations together, we can rewrite 
$$∂E_1/∂a\_h1 = (a\_o1 - t1)  *  a\_o1  *  (1 - a\_o1)  *  w_5$$

Similary the gradients for the error with respect to a_h2 becomes 

$∂E_total/∂a\_h2 = (a\_o1 - t1)  *  a\_o1  *  (1 - a\_o1)  *  w_6 +  (a\_o2 - t2)  *  a\_o2  *  (1 - a\_o2)  *  w_8$

To find the gradients for the weight w1, we can apply the chain rule again

$∂E_total/∂w_1 = ∂E_total/∂a\_h1  *  ∂a\_h1/∂h1  *  ∂h1/∂w_1$

This results in combining all the above into one single equations as 

- $∂E_total/∂w_1 = ((a\_o1 - t1)  *  a\_o1  *  (1 - a\_o1)  *  w_5 +  (a\_o2 - t2)  *  a\_o2  *  (1 - a\_o2)  *  w_7)  *  a\_h1  *  (1 - a\_h1)  *  i_1$

- $∂E_total/∂w_2 = ((a\_o1 - t1)  *  a\_o1  *  (1 - a\_o1)  *  w_5 +  (a\_o2 - t2)  *  a\_o2  *  (1 - a\_o2)  *  w_7)  *  a\_h1  *  (1 - a\_h1)  *  i_2$

- $∂E_total/∂w_3 = ((a\_o1 - t1)  *  a\_o1  *  (1 - a\_o1)  *  w_6 +  (a\_o2 - t2)  *  a\_o2  *  (1 - a\_o2)  *  w_8)  *  a\_h2  *  (1 - a\_h2)  *  i_1$

- $∂E_total/∂w_4 = ((a\_o1 - t1)  *  a\_o1  *  (1 - a\_o1)  *  w_6 +  (a\_o2 - t2)  *  a\_o2  *  (1 - a\_o2)  *  w_8)  *  a\_h2  *  (1 - a\_h2)  *  i_2$


By the calculations from above, we know
$$ a\_o1=0.606477732 , a\_o2=0.630480835 $$ 
$$ t1=0.5 , t2=0.5 $$
$$a\_h1=0.506874567 , a\_h2=0.510623401$$
$$ i1=0.5 , i2=0.05$$

*hence*

$$ ∂E/∂w1=0.000316993 ,∂E/∂w2=0.000633987 $$
$$ ∂E/∂w3=0.000351869, ∂E/∂4=0.000703737 $$
				
### 6. Calculating the new Weights

We have now found the equations using which we can back propogate and adjust the weights. TBased on the learning rate we configure it with, the weights after each pass will be adjusted as 
```
new_weight = old_weight - (learning rate x weight's gradient)
```
The learning rate is a scalar value that determines the step size at which the model's parameters are updated during the optimization process.
The weights get updated based on the above formula. For simplicity, we have taken learning rate as 1. 


## Learning Rate vs Error

Now we are going to see how the error varies based on the learning rate. The following graphs have the X axis as the number of iterations and Y axis as the total error.

We will be considering the following learning rates and see how the error changes [0.1, 0.2, 0.5, 0.8, 1.0, 2.0,1000] 

- For a learning rate =0.1, we see the error decreases linearly with the number of iterations as shown below.

![lr_0_1](./Images/lr_0_1.jpg?raw=true)

- Similary for a learning rate of 0.2, we see somewhat the same.

![lr_0_2](./Images/lr_0_2.jpg?raw=true)

- For learning rate= 0.5, there is a tiny exponential dip on the curve

![lr_0_5](./Images/lr_0_5.jpg?raw=true)


- For learning rate= 0.8, there is a clear evidence of the exponential bend on the error curve

![lr_0_8](./Images/lr_0_8.jpg?raw=true)


- For learning rate= 1, there loss was tending to zero in less number of iterations compared to the learning rate as 0.8

![lr_1](./Images/lr_1.jpg?raw=true)

- For learning rate=2, the loss tending to zero have almost become half the number of iterations to learning rate as 1.

![lr_2](./Images/lr_2.jpg?raw=true)

- For learning rate=1000, the loss is constant after the first iteration and hence does not add any good to the backpropagation.

![lr_1000](./Images/lr_1000.jpg?raw=true)

We can observe that the error approaches zero more quickly as the learning rate rises because the rate of change with each iteration shifts from linear to more exponential. As a result of the weights changing often due to a high learning rate, the ideal point may be overshot or convergence may occur more quickly.
