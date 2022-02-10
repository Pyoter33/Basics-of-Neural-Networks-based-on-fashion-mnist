from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time


def load_mnist(path, kind='train'):
    import os
    import gzip

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                              '%s-labels-idx1-ubyte.gz'
                              % kind)
    images_path = os.path.join(path,
                              '%s-images-idx3-ubyte.gz'
                              % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels


def show_comparison(original, approximation):
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


def show_results(predictions, x_test, y_test):
    labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt',
              'Sneaker', 'Bag', 'Ankle boot']
    chosen_images = np.random.randint(10000, size=6)

    plt.figure(figsize=(10, 8))

    plt.subplot(2, 3, 1)
    plt.imshow(x_test[chosen_images[0]].reshape(28, 28),
               cmap=plt.cm.gray, interpolation='nearest',
               clim=(0, 1))
    plt.xlabel(f'Correct label: {labels[y_test[chosen_images[0]]]}', fontsize=14)
    plt.title(f'Chosen label: {labels[np.argmax(predictions[chosen_images[0]])]}', fontsize=14)

    plt.subplot(2, 3, 2)
    plt.imshow(x_test[chosen_images[1]].reshape(28, 28),
               cmap=plt.cm.gray, interpolation='nearest',
               clim=(0, 1))
    plt.xlabel(f'Correct label: {labels[y_test[chosen_images[1]]]}', fontsize=14)
    plt.title(f'Chosen label: {labels[np.argmax(predictions[chosen_images[1]])]}', fontsize=14)

    plt.subplot(2, 3, 3)
    plt.imshow(x_test[chosen_images[2]].reshape(28, 28),
               cmap=plt.cm.gray, interpolation='nearest',
               clim=(0, 1))
    plt.xlabel(f'Correct label: {labels[y_test[chosen_images[2]]]}', fontsize=14)
    plt.title(f'Chosen label: {labels[np.argmax(predictions[chosen_images[2]])]}', fontsize=14)

    plt.subplot(2, 3, 4)
    plt.imshow(x_test[chosen_images[3]].reshape(28, 28),
               cmap=plt.cm.gray, interpolation='nearest',
               clim=(0, 1))
    plt.xlabel(f'Correct label: {labels[y_test[chosen_images[3]]]}', fontsize=14)
    plt.title(f'Chosen label: {labels[np.argmax(predictions[chosen_images[3]])]}', fontsize=14)

    plt.subplot(2, 3, 5)
    plt.imshow(x_test[chosen_images[4]].reshape(28, 28),
               cmap=plt.cm.gray, interpolation='nearest',
               clim=(0, 1))
    plt.xlabel(f'Correct label: {labels[y_test[chosen_images[4]]]}', fontsize=14)
    plt.title(f'Chosen label: {labels[np.argmax(predictions[chosen_images[4]])]}', fontsize=14)

    plt.subplot(2, 3, 6)
    plt.imshow(x_test[chosen_images[5]].reshape(28, 28),
               cmap=plt.cm.gray, interpolation='nearest',
               clim=(0, 1))
    plt.xlabel(f'Correct label: {labels[y_test[chosen_images[5]]]}', fontsize=14)
    plt.title(f'Chosen label: {labels[np.argmax(predictions[chosen_images[5]])]}', fontsize=14)
    plt.show()


def get_data(train_path, test_path):
    x_train, y_train = load_mnist(train_path, kind='train')
    x_test, y_test = load_mnist(test_path, kind='t10k')

    return x_train, y_train, x_test, y_test


def convert_data(x_train, x_test):
    return x_train / 255, x_test / 255


def reshape_data(x_train, x_test):
    return x_train.reshape((x_train.shape[0], 28, 28, 1)), x_test.reshape((x_test.shape[0], 28, 28, 1))


def apply_pca(x_train, x_test):
    start_time = time.time()
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    pca = PCA(n_components=100)
    lower_dimensional_x_train = pca.fit_transform(x_train_scaled)

    approximation = scaler.inverse_transform(pca.inverse_transform(lower_dimensional_x_train))
    x_test_scaled = scaler.transform(x_test)
    lower_dimensional_x_test = pca.transform(x_test_scaled)

    print(f'Time of PCA: {time.time() - start_time}')
    show_comparison(x_train, approximation)
    return lower_dimensional_x_train, lower_dimensional_x_test


def create_basic_model(shape):
    model = tf.keras.Sequential(
        [tf.keras.layers.InputLayer(input_shape=shape),
         tf.keras.layers.Dense(100, activation='relu'),
         tf.keras.layers.Dense(10, activation='softmax')])

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def create_improved_model():
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


def train_model(x_train, y_train, model, epochs_number):
    model.fit(x_train, y_train, epochs=epochs_number)
    return model


def evaluate_and_predict(x_test, y_test, model):
    test_loss, test_acc = model.evaluate(x_test, y_test)
    predictions = model.predict(x_test)
    return test_loss, test_acc, predictions


def test_models(train_data_path='', test_data_path=''):
    x_train, y_train, x_test, y_test = get_data(train_data_path, test_data_path)

    x_train, x_test = convert_data(x_train, x_test)
    lower_dimensional_x_train, lower_dimensional_x_test = apply_pca(x_train, x_test)

    start_time = time.time()
    print('Training basic model without PCA')
    model = create_basic_model(784)
    model = train_model(x_train, y_train, model, 7)
    test_loss, test_acc, _ = evaluate_and_predict(x_test, y_test, model)
    basic_model_time = time.time() - start_time

    start_time = time.time()
    print('Training basic model with PCA')
    lower_dim_model = create_basic_model(100)
    lower_dim_model = train_model(lower_dimensional_x_train, y_train, lower_dim_model, 7)
    test_loss_pca, test_acc_pca, _ = evaluate_and_predict(lower_dimensional_x_test, y_test, lower_dim_model)
    basic_model_pca_time = time.time() - start_time

    start_time = time.time()
    print('Training improved model')
    reshaped_x_train, reshaped_x_test = reshape_data(x_train, x_test)
    improved_model = create_improved_model()
    improved_model = train_model(reshaped_x_train, y_train, improved_model, 7)
    improved_test_loss, improved_test_acc, predictions = evaluate_and_predict(reshaped_x_test, y_test, improved_model)
    improved_model_time = time.time() - start_time

    print('\n------------------------------------------------------\n')
    print('Basic model')
    print(f'Loss: {test_loss} Accuracy: {test_acc} Time to complete: {basic_model_time} \n')
    print('Basic model with PCA')
    print(f'Loss: {test_loss_pca} Accuracy: {test_acc_pca} Time to complete: {basic_model_pca_time}\n')
    print('Improved model with CNN')
    print(f'Loss: {improved_test_loss} Accuracy: {improved_test_acc} Time to complete: {improved_model_time}')
    show_results(predictions, x_test, y_test)


if __name__ == '__main__':
    test_models()
