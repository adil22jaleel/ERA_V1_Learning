# UNET on Oxford Dataset
This folder contains the UNET implementation on the [Oxford Pets dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/).

## Results

### Max Pooling +Transpose Conv + Binary Cross-Entropy

![image](https://github.com/adil22jaleel/era-v1-assignments/blob/main/s18_assignment/assets/mp_tconv_bce_comparison.png)
```
Epoch 15/20
94/94 [==============================] - 3s 37ms/step - loss: 0.7207 - accuracy: 0.7109 - val_loss: 0.7379 - val_accuracy: 0.7020
Epoch 16/20
94/94 [==============================] - 3s 37ms/step - loss: 0.7159 - accuracy: 0.7133 - val_loss: 0.7309 - val_accuracy: 0.7091
Epoch 17/20
94/94 [==============================] - 3s 37ms/step - loss: 0.7098 - accuracy: 0.7162 - val_loss: 0.7189 - val_accuracy: 0.7113
Epoch 18/20
94/94 [==============================] - 3s 37ms/step - loss: 0.7016 - accuracy: 0.7202 - val_loss: 0.7140 - val_accuracy: 0.7147
Epoch 19/20
94/94 [==============================] - 3s 37ms/step - loss: 0.6895 - accuracy: 0.7256 - val_loss: 0.7191 - val_accuracy: 0.7105
Epoch 20/20
94/94 [==============================] - 3s 37ms/step - loss: 0.6849 - accuracy: 0.7270 - val_loss: 0.7081 - val_accuracy: 0.7170
24/24 [==============================] - 0s 12ms/step - loss: 0.7081 - accuracy: 0.7170
1/1 [==============================] - 1s 704ms/step
```
Sample Prediction:
![image](https://github.com/adil22jaleel/era-v1-assignments/blob/main/s18_assignment/assets/mp_tconv_bce.png)


### Max Pooling + Transpose Conv + Dice Loss

![image](https://github.com/adil22jaleel/era-v1-assignments/blob/main/s18_assignment/assets/mp_tconv_dice_comparison.png)

```
Epoch 15/20
94/94 [==============================] - 3s 36ms/step - loss: 0.7555 - accuracy: 0.5798 - val_loss: 0.7548 - val_accuracy: 0.5826
Epoch 16/20
94/94 [==============================] - 3s 36ms/step - loss: 0.7555 - accuracy: 0.5798 - val_loss: 0.7548 - val_accuracy: 0.5826
Epoch 17/20
94/94 [==============================] - 3s 36ms/step - loss: 0.7555 - accuracy: 0.5798 - val_loss: 0.7548 - val_accuracy: 0.5826
Epoch 18/20
94/94 [==============================] - 3s 36ms/step - loss: 0.7555 - accuracy: 0.5798 - val_loss: 0.7548 - val_accuracy: 0.5826
Epoch 19/20
94/94 [==============================] - 3s 36ms/step - loss: 0.7555 - accuracy: 0.5798 - val_loss: 0.7548 - val_accuracy: 0.5826
Epoch 20/20
94/94 [==============================] - 3s 36ms/step - loss: 0.7555 - accuracy: 0.5798 - val_loss: 0.7548 - val_accuracy: 0.5826
24/24 [==============================] - 0s 12ms/step - loss: 0.7548 - accuracy: 0.5826
1/1 [==============================] - 0s 333ms/step
```
Sample prediction:
![image](https://github.com/adil22jaleel/era-v1-assignments/blob/main/s18_assignment/assets/mp_tconv_dice.png)

### Strided Convolution + Transpose Conv + Binary Cross Entropy
![image](https://github.com/adil22jaleel/era-v1-assignments/blob/main/s18_assignment/assets/sconv_tconv_bce_comparison.png)

```
Epoch 15/20
94/94 [==============================] - 4s 40ms/step - loss: 0.7420 - accuracy: 0.6994 - val_loss: 0.7461 - val_accuracy: 0.6994
Epoch 16/20
94/94 [==============================] - 4s 40ms/step - loss: 0.7364 - accuracy: 0.7025 - val_loss: 0.7375 - val_accuracy: 0.7010
Epoch 17/20
94/94 [==============================] - 4s 40ms/step - loss: 0.7326 - accuracy: 0.7039 - val_loss: 0.7351 - val_accuracy: 0.7037
Epoch 18/20
94/94 [==============================] - 4s 39ms/step - loss: 0.7276 - accuracy: 0.7066 - val_loss: 0.7342 - val_accuracy: 0.7022
Epoch 19/20
94/94 [==============================] - 4s 40ms/step - loss: 0.7246 - accuracy: 0.7073 - val_loss: 0.7329 - val_accuracy: 0.7026
Epoch 20/20
94/94 [==============================] - 4s 40ms/step - loss: 0.7235 - accuracy: 0.7080 - val_loss: 0.7340 - val_accuracy: 0.7048
24/24 [==============================] - 0s 13ms/step - loss: 0.7340 - accuracy: 0.7048
1/1 [==============================] - 0s 415ms/step
```
Sample Prediction:
![image](https://github.com/adil22jaleel/era-v1-assignments/blob/main/s18_assignment/assets/sconv_tconv_bce.png)

### Strided Conv + Up Sampling + Dice Loss

![image](https://github.com/adil22jaleel/era-v1-assignments/blob/main/s18_assignment/assets/sconv_sampling_dice_comparison.png)

```
Epoch 15/20
94/94 [==============================] - 4s 42ms/step - loss: 0.7555 - accuracy: 0.5798 - val_loss: 0.7548 - val_accuracy: 0.5826
Epoch 16/20
94/94 [==============================] - 4s 42ms/step - loss: 0.7555 - accuracy: 0.5798 - val_loss: 0.7548 - val_accuracy: 0.5826
Epoch 17/20
94/94 [==============================] - 4s 42ms/step - loss: 0.7555 - accuracy: 0.5798 - val_loss: 0.7548 - val_accuracy: 0.5826
Epoch 18/20
94/94 [==============================] - 4s 42ms/step - loss: 0.7555 - accuracy: 0.5798 - val_loss: 0.7548 - val_accuracy: 0.5826
Epoch 19/20
94/94 [==============================] - 4s 43ms/step - loss: 0.7555 - accuracy: 0.5798 - val_loss: 0.7548 - val_accuracy: 0.5826
Epoch 20/20
94/94 [==============================] - 4s 42ms/step - loss: 0.7555 - accuracy: 0.5798 - val_loss: 0.7548 - val_accuracy: 0.5826
24/24 [==============================] - 0s 15ms/step - loss: 0.7548 - accuracy: 0.5826
1/1 [==============================] - 0s 302ms/step
```
Sample Prediction:
![image](https://github.com/adil22jaleel/era-v1-assignments/blob/main/s18_assignment/assets/sconv_sampling_dice.png)
