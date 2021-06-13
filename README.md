# Fashion-MNIST

## Introduction
The Fashion-MNIST database is a great tool for practicing various machine learning algorithms. The database consists of a training set of 60000 and a test set of 10000 examples of different pieces of clothing associated with 10 different classes. The examples are represented by 28x28 grayscale images. The goal is to create an algorithm which will be able to achieve the highest accuracy of classifying images from the test set to their respective labels. To get to that point the algorithm has to be first trained with the provided training dataset. There many ways to properly train the algorithm. I have decided to use neural networks.

## Methods
### Data preprocessing
#### Basic preprocessing
If every image has a 28x28 resolution, this means that it is represented by 784 features (pixels). Every feature has a value between 0 (completely black) and 255 (completely white). Neural networks work better when the values provided to the input layer have relatively small and standardized values e.g. [0, 1] or [-1, 1]. So the logical thing to do would be to divide every value by 255. This provides us with a list of features with values between 0 and 1. 

```python
  xTrain / 255
  xTest / 255
```
#### PCA (Principal Component Analysis)
PCA is used for dimensionality reduction for the visualization of high dimensional data, which means it is used to reduce number of features representing certain piece of data, in this case - image. PCA operates by finding the Principal Components of data which are  the directions, where there is the most variance in the data. Than it projects it into smaller dimensional subspace while maintaining most of the information. Detailed explanation of how the PCA works is available at 
https://towardsdatascience.com/the-mathematics-behind-principal-component-analysis-fff2d7f4b643. 

The sklearn library provides PCA class which takes care of the advanced mathematics for us. To further improve the effectiveness of the PCA it is important to use a StandardScaler first to standardize the dataset features even further. 
```python
  scaler = StandardScaler()
  xTrainScaled = scaler.fit_transform(xTrain)
  pca = PCA(n_components=100)
  lowerDimensionalXTrain = pca.fit_transform(xTrainScaled)

  approximation = scaler.inverse_transform(pca.inverse_transform(lowerDimensionalXTrain))
  xTestScaled = scaler.transform(xTest)
  lowerDimensionalXTest = pca.transform(xTestScaled)
```
It is crucial to fit and transform training data and only transform test data. If we were to use fit method on the test data too, new mean and variance would be computed which would lead the model to also learning this data and thus making it impossible to properly predict the model’s efficiency.

![pcaVisualization](https://user-images.githubusercontent.com/84713157/121813407-31dc7d80-cc6c-11eb-8149-2188c64f6dbd.png)
*Visualization of the loss of quality after applying PCA*

### Creating models
#### Neural networks
Neural network is a layered representation of data vaguely inspired by the biological neural networks that constitute animal brains. Each layer consists of nodes called neurons. The nodes are connected with each other with edges. Edges typically have weights, which can be adjusted in the training process. Each layer also has a bias – an extra neuron that has no connections and holds a single numeric value. 
Every neural network has an input layer, output layer and can have a number of hidden layers. The input layer is the layer that initial data is passed to. It is the first layer in every neural network. The output layer is the layer that we will retrieve our results from. All the other layers in the neural network are called hidden layers. The name ‘hidden’ represents the fact that their data cannot be observed like in case of output and input layers.
The neurons at each layer are used to pass and transform data. The data at each subsequent neuron is defined by a weighted sum  with added bias. 

![equasion](https://user-images.githubusercontent.com/84713157/121813663-5553f800-cc6d-11eb-995b-55b6e60958ed.png)

w - weight of connection to the neuron

x - value of connected neuron

b - bias

To add more complexity and dimensionality to the network, it is important to apply an activation function to the mentioned equation. There are no restrictions about which function to use, but higher order/degree functions like sigmoid, ReLU or softmax are definitely best suited for this task.

![Inkedneuralnetwork2](https://user-images.githubusercontent.com/84713157/121813895-779a4580-cc6e-11eb-9479-3208f0fc4bde.jpg)

*Visualization of one neuron*


#### Cost function
Cost function determines how close the model is from correct classifications by comparing model outputs with expected outputs provided with the training data set. 

#### Backpropagation
Backpropagation is a process of changing weights and biases in a neural network. It calculates the gradient of the cost function which than can be used to change weights and biases to improve the model.

#### Convolutional Neural Networks (CNN)
Convolutional Neural Networks are great when it comes to the problem of image classification. Contrary to the basic neural network, the CNN can be trained to match certain patterns from the images. Adding this to the basic model can visibly improve its efficiency.
The patterns are created with filters/kernels that are small matrices. The filter is applied to the input matrix using the convolution operation. It creates an output matrix which can be considered as a kind of a map of where the certain pattern occurs on the image. Using more filters in the convolutional network produces more maps of patterns which can than improve the classification. 

![convolution](https://user-images.githubusercontent.com/84713157/121814070-6bfb4e80-cc6f-11eb-8c70-0a29476f6c93.png)

*Example of convolution*


After the convolution it is a common practice to add pooling to the model. Pooling shrinks the matrix representing the image by taking maximum values from small windows which are ‘walked over’ the current matrix. The result is a smaller matrix but it still represents the given pattern. A smaller data is convenient for the neural networks and also makes the algorithm less sensitive which in the long run prevents overfitting.

![pooling](https://user-images.githubusercontent.com/84713157/121814117-b381da80-cc6f-11eb-97c8-13d84eb7795f.png)

*Example of pooling*


### Implementing models
#### Basic model 
The tensorflow library greatly simplifies creating neural networks. The first network I created is a basic sequential model with one hidden layer. The activation functions are ReLU on the hidden layer and softmax on the output. The model is using sparse categorical crossentropy as the loss function. This function produces a category index of the most likely matching category. The optimizer – adam – is a stochastic gradient descent method. 
```python
    model = tf.keras.Sequential(
        [tf.keras.layers.InputLayer(input_shape=shape),
         tf.keras.layers.Dense(100, activation='relu'),
         tf.keras.layers.Dense(10, activation='softmax')])

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
```

#### Improved model
In the improved model instead of the basic input layer there is a convolution layer. It has 32 filters. This number works fine for datasets which are not very complicated. The kernel size 3x3 is a suitable number for low dimensional images such as the clothing images used in MNIST database. In this model I am also using different kernel initializer. Kernel initializers are functions that determine the starting values of weights used in neural networks. Chosen he uniform initializer is better for nodes which use ReLU activation functions. The second layer is also convolutional but this time it has 64 filters as it is deeper in the network.  The next layer in the model applies the previously mentioned pooling with the window size of 2x2. Rest of the model looks similar to the basic one but with a smaller number of neurons in the hidden layer.
```python
    xTrain.reshape((xTrain.shape[0], 28, 28, 1)) 
    xTest.reshape((xTest.shape[0], 28, 28, 1))
  
    model = tf.keras.Sequential(
        [tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform',
                                input_shape=(28, 28, 1)),
         tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'),
         tf.keras.layers.MaxPooling2D((2, 2)),
         tf.keras.layers.Flatten(),
         tf.keras.layers.Dense(70, activation='relu', kernel_initializer='he_uniform'),
         tf.keras.layers.Dense(10, activation='softmax')])

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
```

#### Training and evaluating
The model is trained in epochs. Epochs determine how many times will the model be trained on the same training data set. The higher the number, the more accurate the predictions are. But if the number of epochs is too high the risk of overfitting is much greater and the training process will of course be longer. I have decided that 7 epochs will be suitable for this task.
To determine the accuracy of the algorithm it has to be evaluated. Evaluation returns the final accuracy and loss based on comparing model outputs with correct labels in the test dataset. Predictions return an array of probabilities of every labels for every image. It can be than used to create various visualizations.
```python
    model.fit(xTrain, yTrain, epochs=epochsNumber)
    
    testLoss, testAcc = model.evaluate(xTest, yTest)
    predictions = model.predict(xTest)
```

### Visualization
![mist1](https://user-images.githubusercontent.com/84713157/121816622-fa2a0180-cc7c-11eb-86c8-e11c50d65719.png)

![mnist3](https://user-images.githubusercontent.com/84713157/121816627-feeeb580-cc7c-11eb-9ec9-d625cdf56e81.png)

![mnist2](https://user-images.githubusercontent.com/84713157/121816625-fc8c5b80-cc7c-11eb-8976-339b48a99401.png)


## Results

### Tests
First I tested the database with the KNN algorithm which I wrote for one of the prievous tasks with PCA applied on the dataset. The algorithm concluded that it can reach a smallest classification error at around 0,25 with 200 neighbours. 

For this task I have decided to test three algorithms. The first one is a basic machine learning model that I mentioned earlier, the second is the same model but the dimensionality of data provided for it is reduced by the PCA to 100 features and the last one which uses convolutional neural network. In the tests I am also using computing power of my computer's graphics card.

**Computer used for tests:**

CPU: AMD Ryzen 7 3700X 8 cores 3,6 GHz

Graphics: NVIDIA GeForce RTX 2060 SUPER 8GB

RAM: 32GB

**Basic model**
|Test | Acuracy |Loss | Time |
|---|---|---|---|
|1. |0,869|0,362|24,86s|
|2. |0,875|0,350|23,93s|
|3. |0,875|0,349|23,68s|
|4. |0,877|0,336|23,83s|
|5. |0,879|0,340|23,21s|
|AVG|0,875|0,347|23,90s|

**Basic model with PCA**

AVG PCA time: 4,10s

Time of PCA convertion is not included in the table!

|Test | Acuracy |Loss | Time |
|---|---|---|---|
|1. |0,875|0,356|23,08s|
|2. |0,876|0,364|21,54s|
|3. |0,876|0,361|21,28s|
|4. |0,877|0,351|21,99s|
|5. |0,874|0,365|21,41s|
|AVG|0,876|0,359|21,86s|

**Improved model with CNN**
|Test | Acuracy |Loss | Time |
|---|---|---|---|
|1. |0,916|0,306|42,95s|
|2. |0,919|0,342|46,85s|
|3. |0,921|0,304|43,05s|
|4. |0,919|0,342|42,56s|
|5. |0,924|0,291|46,27s|
|AVG|0,920|0,317|44,34s|

**Note**

Running the tests without the graphic processor's support gave much different time results. Basic models were approximately 3 times faster but the improved model trained the data more than 6 times longer. As expected, other results were similar.


### Comparison with similar models

To compare my model I have put it along with other which also use CNN.

|Submiter|Method|Accuracy|
|---|---|---|
|Me|2 Conv + pooling + hidden layer| 0,920|
|Kashif Rasul|2 Conv+pooling|0,876|
|Tensorflow's doc|2 Conv+pooling|0,916|
|@AbhirajHinge|2 Conv+pooling+ELU activation (PyTorch)|0,903|
|Kyriakos Efthymiadis|2 Conv with extensive preprocessing|0,919|
|@khanguyen1207|2 Conv+pooling+batch normalization|0,934|

## Usage

### Packages and imports
Required packages:
* tensorflow 2.6.0 – for creating and testing neural networks
* sklearn – for implementing PCA
* matplotlib – for creating visualizations
* numpy – for matrix operations
* time – for time evaluation

**Imports:**
```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time
```
### Running

To run the script you need to download the code from github repository. If you have the mnist dataset files in the same folder as the program you do not need to set any path. If you have it in another directory, you need to specify them as arguments in the method testModels() in main giving path to the directory with training files first and the one with test files second. To start training and see visualizations and statistics just run the main function.

To get similar time results to those showed in the Results section you need to have a graphics card supported by NVIDIA CUDA. Than you need to install CUDA on your computer. Instruction: https://developer.nvidia.com/cuda-zone

It may be possible for tensorflow to inform you about missing libraries. You can get them from the cuDNN library. To be able to download it you need to join the NVIDIA Developer Program. Info: https://developer.nvidia.com/cudnn

## Sources
* CNN: https://youtu.be/FmpDIaiMIeA
* Machine learning and neural networks: https://youtu.be/tPYj3fFJGjk
* CNN: https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53
* PCA: https://towardsdatascience.com/what-and-why-behind-fit-transform-vs-transform-in-scikit-learn-78f915cf96fe
* Optimizers: https://towardsdatascience.com/optimizers-for-training-neural-network-59450d71caf6
* CNN: https://www.pyimagesearch.com/2018/12/31/keras-conv2d-and-convolutional-layers/
* PCA: https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
* PCA: https://www.kaggle.com/residentmario/dimensionality-reduction-and-pca-for-fashion-mnist
* Feature extraction: https://towardsdatascience.com/image-feature-extraction-traditional-and-deep-learning-techniques-ccc059195d04
* TensorFlow Docs
* Scikit Docs
