from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time


def loadMnist(path, kind='train'):
    import os
    import gzip

    """Load MNIST data from `path`"""
    labelsPath = os.path.join(path,
                              '%s-labels-idx1-ubyte.gz'
                              % kind)
    imagesPath = os.path.join(path,
                              '%s-images-idx3-ubyte.gz'
                              % kind)

    with gzip.open(labelsPath, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(imagesPath, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels


def showComparison(original, approximation):
    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(original[6].reshape(28, 28),
               cmap=plt.cm.gray, interpolation='nearest',
               clim=(0, 1))
    plt.xlabel('784 components', fontsize=14)
    plt.title('Original Image', fontsize=20)

    plt.subplot(1, 2, 2)
    plt.imshow(approximation[6].reshape(28, 28),
               cmap=plt.cm.gray, interpolation='nearest',
               clim=(0, 1))
    plt.xlabel('100 components', fontsize=14)
    plt.title('Transformed image', fontsize=20)
    plt.show()


def showResults(predictions, xTest, yTest):
    labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt',
              'Sneaker', 'Bag', 'Ankle boot']
    chosenImages = np.random.randint(10000, size=6)

    plt.figure(figsize=(10, 8))

    plt.subplot(2, 3, 1)
    plt.imshow(xTest[chosenImages[0]].reshape(28, 28),
               cmap=plt.cm.gray, interpolation='nearest',
               clim=(0, 1))
    plt.xlabel(f'Correct label: {labels[yTest[chosenImages[0]]]}', fontsize=14)
    plt.title(f'Chosen label: {labels[np.argmax(predictions[chosenImages[0]])]}', fontsize=14)

    plt.subplot(2, 3, 2)
    plt.imshow(xTest[chosenImages[1]].reshape(28, 28),
               cmap=plt.cm.gray, interpolation='nearest',
               clim=(0, 1))
    plt.xlabel(f'Correct label: {labels[yTest[chosenImages[1]]]}', fontsize=14)
    plt.title(f'Chosen label: {labels[np.argmax(predictions[chosenImages[1]])]}', fontsize=14)

    plt.subplot(2, 3, 3)
    plt.imshow(xTest[chosenImages[2]].reshape(28, 28),
               cmap=plt.cm.gray, interpolation='nearest',
               clim=(0, 1))
    plt.xlabel(f'Correct label: {labels[yTest[chosenImages[2]]]}', fontsize=14)
    plt.title(f'Chosen label: {labels[np.argmax(predictions[chosenImages[2]])]}', fontsize=14)

    plt.subplot(2, 3, 4)
    plt.imshow(xTest[chosenImages[3]].reshape(28, 28),
               cmap=plt.cm.gray, interpolation='nearest',
               clim=(0, 1))
    plt.xlabel(f'Correct label: {labels[yTest[chosenImages[3]]]}', fontsize=14)
    plt.title(f'Chosen label: {labels[np.argmax(predictions[chosenImages[3]])]}', fontsize=14)

    plt.subplot(2, 3, 5)
    plt.imshow(xTest[chosenImages[4]].reshape(28, 28),
               cmap=plt.cm.gray, interpolation='nearest',
               clim=(0, 1))
    plt.xlabel(f'Correct label: {labels[yTest[chosenImages[4]]]}', fontsize=14)
    plt.title(f'Chosen label: {labels[np.argmax(predictions[chosenImages[4]])]}', fontsize=14)

    plt.subplot(2, 3, 6)
    plt.imshow(xTest[chosenImages[5]].reshape(28, 28),
               cmap=plt.cm.gray, interpolation='nearest',
               clim=(0, 1))
    plt.xlabel(f'Correct label: {labels[yTest[chosenImages[5]]]}', fontsize=14)
    plt.title(f'Chosen label: {labels[np.argmax(predictions[chosenImages[5]])]}', fontsize=14)
    plt.show()


def getData(trainPath, testPath):
    xTrain, yTrain = loadMnist(trainPath, kind='train')
    xTest, yTest = loadMnist(testPath, kind='t10k')

    return xTrain, yTrain, xTest, yTest


def convertData(xTrain, xTest):
    return xTrain / 255, xTest / 255


def reshapeData(xTrain, xTest):
    return xTrain.reshape((xTrain.shape[0], 28, 28, 1)), xTest.reshape((xTest.shape[0], 28, 28, 1))


def applyPCA(xTrain, xTest):
    startTime = time.time()
    scaler = StandardScaler()
    xTrainScaled = scaler.fit_transform(xTrain)
    pca = PCA(n_components=100)
    lowerDimensionalXTrain = pca.fit_transform(xTrainScaled)

    approximation = scaler.inverse_transform(pca.inverse_transform(lowerDimensionalXTrain))
    xTestScaled = scaler.transform(xTest)
    lowerDimensionalXTest = pca.transform(xTestScaled)

    print(f'Time of PCA: {time.time() - startTime}')
    showComparison(xTrain, approximation)
    return lowerDimensionalXTrain, lowerDimensionalXTest


def createBasicModel(shape):
    model = tf.keras.Sequential(
        [tf.keras.layers.InputLayer(input_shape=shape),
         tf.keras.layers.Dense(100, activation='relu'),
         tf.keras.layers.Dense(10, activation='softmax')])

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def createImprovedModel():
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
    return model


def trainModel(xTrain, yTrain, model, epochsNumber):
    model.fit(xTrain, yTrain, epochs=epochsNumber)
    return model


def evaluateAndPredict(xTest, yTest, model):
    testLoss, testAcc = model.evaluate(xTest, yTest)
    predictions = model.predict(xTest)
    return testLoss, testAcc, predictions


def testModels(trainDataPath='', testDataPath=''):
    xTrain, yTrain, xTest, yTest = getData(trainDataPath, testDataPath)

    xTrain, xTest = convertData(xTrain, xTest)
    lowerDimensionalXTrain, lowerDimensionalXTest = applyPCA(xTrain, xTest)

    startTime = time.time()
    print('Training basic model without PCA')
    model = createBasicModel(784)
    model = trainModel(xTrain, yTrain, model, 7)
    testLoss, testAcc, _ = evaluateAndPredict(xTest, yTest, model)
    basicModelTime = time.time() - startTime

    startTime = time.time()
    print('Training basic model with PCA')
    lowerDimModel = createBasicModel(100)
    lowerDimModel = trainModel(lowerDimensionalXTrain, yTrain, lowerDimModel, 7)
    testLossPCA, testAccPCA, _ = evaluateAndPredict(lowerDimensionalXTest, yTest, lowerDimModel)
    basicModelPCATime = time.time() - startTime

    startTime = time.time()
    print('Training improved model')
    reshapedXTrain, reshapedXTest = reshapeData(xTrain, xTest)
    improvedModel = createImprovedModel()
    improvedModel = trainModel(reshapedXTrain, yTrain, improvedModel, 7)
    improvedTestLoss, improvedTestAcc, predictions = evaluateAndPredict(reshapedXTest, yTest, improvedModel)
    improvedModelTime = time.time() - startTime

    print('\n------------------------------------------------------\n')
    print('Basic model')
    print(f'Loss: {testLoss} Accuracy: {testAcc} Time to complete: {basicModelTime} \n')
    print('Basic model with PCA')
    print(f'Loss: {testLossPCA} Accuracy: {testAccPCA} Time to complete: {basicModelPCATime}\n')
    print('Improved model with CNN')
    print(f'Loss: {improvedTestLoss} Accuracy: {improvedTestAcc} Time to complete: {improvedModelTime}')
    showResults(predictions, xTest, yTest)


if __name__ == '__main__':
    testModels()
