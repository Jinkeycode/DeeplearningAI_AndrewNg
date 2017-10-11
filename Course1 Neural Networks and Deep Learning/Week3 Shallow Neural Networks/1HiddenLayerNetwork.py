# -*- coding: utf-8 -*-
__author__ = 'Jinkey'
'''
遇到的坑：
（1）因为np.random.seed(2)的位置不同，导致初始化的值不同。在类初始化前写和在初始胡 w1的时候写的结果不同
（2）一开始学习率设置为0.005导致在训练太慢，迭代10000次之后，在训练集准确率只有60%多；而课程的作业是把学习率设置成了1.2 。说明学习率的区别导致巨大的训练效果差异。
'''
# Package imports
import numpy as np
import matplotlib.pyplot as plt
from testCases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
import h5py
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

class OneHiddenLayerNetwork:
    def __init__(self):
        self.X, self.Y = load_planar_dataset()
        self.sample_amount = self.X.shape[1]
        self.n_x = self.X.shape[0]
        self.n_h = 4
        self.n_y = self.Y.shape[0]
        np.random.seed(2)
        self.W1 = np.random.randn(self.n_h, self.n_x) * 0.01 # (后一层，前一层)
        print self.W1
        self.b1 = np.zeros((self.n_h, 1))
        self.W2 = np.random.randn(self.n_y, self.n_h) * 0.01
        print self.W2
        self.b2 = np.zeros((self.n_y, 1))
        self.A1 = np.zeros((self.n_h, self.sample_amount))
        self.A2 = np.zeros((self.n_y, self.sample_amount))
        self.current_cost = 0
        self.is_trained = False

    def print_data_info(self):
        print("其中一个样本：", self.X[:, 0])
        print("维度：%d， 样本数：%d" % (self.X.shape[0], self.X.shape[1]))
        print(self.Y.shape)
        return self

    def show_scatter(self):
        print(np.squeeze(self.Y).shape)
        print(self.Y.shape)
        plt.scatter(self.X[0, :], self.X[1, :], c=np.squeeze(self.Y), s=20, cmap=plt.cm.Spectral)
        plt.show()
        return self

    def simple_lr(self):
        clf = sklearn.linear_model.LogisticRegressionCV()
        clf.fit(self.X.T, np.squeeze(self.Y))
        print self.X.T.shape
        plot_decision_boundary(lambda x: clf.predict(x), self.X, np.squeeze(self.Y))
        plt.title("Logistic Regression")
        plt.show()
        LR_predictions = clf.predict(self.X.T)
        print ('Accuracy of logistic regression: %d ' % float(
            (np.dot(self.Y, LR_predictions) + np.dot(1 - self.Y, 1 - LR_predictions)) / float(self.Y.size) * 100) +
               '% ' + "(percentage of correctly labelled datapoints)")
        return self

    def validate(self):
        assert (self.W1.shape == (self.n_h, self.n_x))
        assert (self.b1.shape == (self.n_h, 1))
        assert (self.W2.shape == (self.n_y, self.n_h))
        assert (self.b2.shape == (self.n_y, 1))
        assert (self.X.shape[1] == self.Y.shape[1] == self.sample_amount)
        assert (self.A1.shape == (self.n_h, self.sample_amount))
        assert (self.A2.shape == (self.n_y, self.sample_amount))
        return self

    def forward_propagation(self):
        Z1 = np.dot(self.W1, self.X) + self.b1
        self.A1 = np.tanh(Z1)

        Z2 = np.dot(self.W2, self.A1) + self.b2
        self.A2 = sigmoid(Z2)
        self.validate()

        cost = - 1.0 / self.sample_amount * np.sum(np.log(self.A2)*self.Y + (1-self.Y)*np.log(1-self.A2))
        self.current_cost = np.squeeze(cost)

        return self

    def backward_propagation(self, learning_rate):
        dZ2 = self.A2 - self.Y
        dW2 = 1.0 / self.sample_amount * np.dot(dZ2, self.A1.T)  # (1, 4) = (1, 400) * (4, 400).T
        db2 = 1.0 / self.sample_amount * np.sum(dZ2, axis=1, keepdims=True)

        dZ1 = np.dot(self.W2.T, dZ2) * (1 - np.power(self.A1, 2))  # (1, 4) * (1, 400) * (4, 400)
        dW1 = 1.0 / self.sample_amount * np.dot(dZ1, self.X.T)
        db1 = 1.0 / self.sample_amount * np.sum(dZ1, axis=1, keepdims=True)

        self.W1 = self.W1 - learning_rate * dW1
        print(self.W1)
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.validate()

    def save_parameters(self):

        self.validate()

        f = h5py.File("1nn.h5", "w")
        f.create_dataset("W1", data=self.W1)
        f.create_dataset("b1", data=self.b1)
        f.create_dataset("W2", data=self.W2)
        f.create_dataset("b2", data=self.b2)

    def load_parameters(self):
        f = h5py.File("1nn.h5", "r")
        self.W1 = np.array(f["W1"])
        self.b1 = np.array(f["b1"])
        self.W2 = np.array(f["W2"])
        self.b2 = np.array(f["b2"])

        self.validate()

        self.is_trained = True
        return self

    def train(self, num_iterations=10000, learning_rate=1.2):
        for i in range(0, num_iterations):
            self.forward_propagation()
            self.backward_propagation(learning_rate=learning_rate)

            if i % 1000 == 0:
                print("Cost after iteration %i: %f" % (i, self.current_cost))

        self.save_parameters()
        return self

    def predict(self, x):
        assert self.is_trained
        print(x.shape)
        assert x.shape == (2, x.shape[1])

        Z1 = np.dot(self.W1, x) + self.b1
        A1 = np.tanh(Z1)
        Z2 = np.dot(self.W2, A1) + self.b2
        A2 = sigmoid(Z2)


        return A2

    def show_decision_boundary(self):
        predict_array = self.predict(self.X)
        predict_array[predict_array > 0.5] = 1
        predict_array[predict_array <= 0.5] = 0

        compare_array = predict_array - self.Y  # (1, 400) - (1, 400)
        print(compare_array)
        accuracy = len(compare_array[compare_array == 0])/float(self.sample_amount)
        print("在训练集上的准确率为：{}".format(accuracy))

        print(self.X.shape)
        print(self.Y.shape)
        plot_decision_boundary(lambda x: self.predict(x.T), self.X, np.squeeze(self.Y))
        plt.title("One hidden layer network")
        plt.show()


if __name__ == '__main__':
    np.random.seed(2)
    W1 = np.random.randn(4, 2) * 0.01
    W2 = np.random.randn(1, 4) * 0.01

    # print(W1)
    # print(W2)
    # 试试逻辑回归
    # OneHiddenLayerNetwork().print_data_info().simple_lr()
    # 训练
    # OneHiddenLayerNetwork().validate().train(num_iterations=10000, learning_rate=1.2)
    # 显示模型效果
    OneHiddenLayerNetwork().load_parameters().show_decision_boundary()
    # 预测一个样本
