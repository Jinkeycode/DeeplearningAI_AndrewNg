# -*- coding: utf-8 -*-
__author__ = 'Jinkey'

import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from dnn_app_utils import *
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation

np.random.seed(1)


class DeepClassifier:
    def __init__(self):
        self.train_x, self.train_y, self.test_x, self.test_y, self.classes = load_data()
        self.sample_amount = self.train_x.shape[0]
        self.test_amount = self.test_x.shape[0]
        self.model = Sequential()
        self.parameters = {}
        self.is_trained = False

    def show_data_info(self):
        print ("Number of training examples: " + str(self.sample_amount))
        print ("Number of testing examples: " + str(self.test_amount))
        print ("Each image is of size: (" + str(self.train_x.shape[1]) + ", " + str(self.train_x.shape[1]) + ", 3)")
        print ("train_x_orig shape: " + str(self.train_x.shape))
        print ("train_y shape: " + str(self.train_y.shape))
        print ("test_x_orig shape: " + str(self.test_x.shape))
        print ("test_y shape: " + str(self.test_y.shape))
        return self

    def flattern_x(self):
        self.train_x = self.train_x.reshape(self.sample_amount, -1).T
        self.test_x = self.test_x.reshape(self.test_amount, -1).T
        assert self.train_x.shape == (12288, self.sample_amount)
        assert self.test_x.shape == (12288, self.test_amount)
        return self

    def standardize_x(self):
        self.train_x = self.train_x / 255.0
        self.test_x = self.test_x / 255.0
        return self

    def L_layer_model(self, learning_rate=0.0075, num_iterations=3000):  # lr was 0.009

        np.random.seed(1)
        costs = []  # keep track of cost

        # Parameters initialization.
        ### START CODE HERE ###
        layers_dims = [12288, 20, 7, 5, 1]
        parameters = initialize_parameters_deep(layers_dims)
        ### END CODE HERE ###

        # Loop (gradient descent)
        for i in range(0, num_iterations):

            # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
            ### START CODE HERE ### (≈ 1 line of code)
            AL, caches = L_model_forward(self.train_x, parameters)
            ### END CODE HERE ###

            # Compute cost.
            ### START CODE HERE ### (≈ 1 line of code)
            cost = compute_cost(AL, self.train_y)
            ### END CODE HERE ###

            # Backward propagation.
            ### START CODE HERE ### (≈ 1 line of code)
            grads = L_model_backward(AL, self.train_y, caches)
            ### END CODE HERE ###

            # Update parameters.
            ### START CODE HERE ### (≈ 1 line of code)
            self.parameters = update_parameters(parameters, grads, learning_rate)
            ### END CODE HERE ###

            # Print the cost every 100 training example
            if i % 100 == 0:
                costs.append(cost)
                print ("Cost after iteration %i: %f" % (i, cost))

        self.is_trained = True

        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        return self

    def save_model(self):
        f = h5py.File("iscat-deep.h5", "w")
        f.create_dataset("layers", data=5)
        for key, value in self.parameters.items():
            f.create_dataset(key, data=value)

    def load_model(self):
        f = h5py.File("iscat-deep.h5", "r")
        number_of_layers = np.squeeze(f["layers"])
        for i in range(1, number_of_layers):
            self.parameters["W"+str(i)] = np.array(f["W"+str(i)])
            self.parameters["b"+str(i)] = np.array(f["b"+str(i)])

        self.is_trained = True
        return self

    # 课程作业也是只是呈现到多层网络的前向和反向传播，前馈和反馈的函数都封装在dnn_app_utils 里面，所以这里我直接用 keras 实现课程作业要求的5层神经网络
    def L_layer_model_with_keras(self):
        model = Sequential()
        model.add(Dense(output_dim=20, activation="relu", input_dim=12288))
        model.add(Dense(output_dim=7, activation="relu", input_dim=13))
        model.add(Dense(output_dim=5, activation="relu", input_dim=7))
        model.add(Dense(output_dim=1, activation="sigmoid", input_dim=5))

        model.compile(loss="binary_crossentropy", optimizer='sgd', metrics=["accuracy"])
        model.fit(self.train_x.T, self.train_y.T, nb_epoch=5000)
        model.save("iscat-keras.h5")
        score = model.evaluate(self.test_x.T, self.test_y.T)

        print(score)
        return self

    def load_keras_model(self):
        self.model = load_model('iscat-keras.h5')
        return self

    def predict_with_keras(self, image_path):
        image = np.array(ndimage.imread(image_path, flatten=False))
        image_flatten = scipy.misc.imresize(image, size=(64, 64)).reshape((64*64*3, 1))
        result = np.squeeze(self.model.predict(image_flatten.T))
        print("这%s一只猫" % "是" if result==1 else "不是")

    def predict_standard(self, image_path):
        print("==============在测试集的准确率=================")
        predict(self.test_x, self.test_y, self.parameters)
        print("==============预测一张图片=================")
        image = np.array(ndimage.imread(image_path, flatten=False))
        my_image = scipy.misc.imresize(image, size=(64, 64)).reshape((64 * 64 * 3, 1))
        my_predicted_image = predict(X=my_image, y=[1], parameters=self.parameters)
        print("这%s一只猫" % "是" if my_predicted_image == 1 else "不是")
        plt.imshow(image)


if __name__ == '__main__':
    # 使用作业方法训练模型
    # DeepClassifier().flattern_x().standardize_x().L_layer_model(learning_rate=0.0075, num_iterations=3000).save_model()
    # 使用 作业的 模型预测
    DeepClassifier().load_model().flattern_x().standardize_x().predict_standard("images/cat.jpg")

    # 使用 Keras 训练模型
    # DeepClassifier().flattern_x().standardize_x().L_layer_model_with_keras()
    # 使用 Keras 模型预测
    # DeepClassifier().load_model().predict("images/cat.jpg")

