# -*- coding: utf-8 -*-
__author__ = 'Jinkey'

import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from  PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

class LogisticRegression:

    def __init__(self, weight_save_path="iscat.h5"):
        self.train_X, self.train_Y, self.test_X, self.test_Y, self.classification = load_dataset()
        self.sample_amount = self.train_X.shape[0]
        self.image_dim = np.prod(self.train_X.shape[1:4])
        self.w = np.zeros((self.image_dim, 1), dtype=float)
        self.b = 0
        self.is_trained = False
        self.weight_save_path = weight_save_path

    def show_data_info(self):
        print("训练样本数：%s" %  str(self.sample_amount))
        print("训练集输入值行列数：%s" %  str(self.train_X.shape))
        print("训练集输出值行列数：%s" % str(self.train_Y.shape))
        print("分类种类数：%s" % str(self.classification.shape))

        self.sample_amount

        plt.imshow(self.train_X[0])  # 非猫示例
        # plt.waitforbuttonpress()
        plt.imshow(self.train_X[27])  # 猫示例
        # plt.waitforbuttonpress()

        return self

    @staticmethod
    def __faltten_three_channel_image_daata(X, number):
        X = np.array(X)
        assert X.shape == (number, 64, 64, 3)
        flatten_X = X.reshape(X.shape[0], X.shape[1]*X.shape[2]*X.shape[3])
        assert flatten_X.shape == (number, 12288)
        return flatten_X

    def faltten_x(self):
        self.train_X = self.__faltten_three_channel_image_daata(self.train_X, 209)
        self.test_X = self.__faltten_three_channel_image_daata(self.test_X, 50)
        return self

    def standardize_x(self):
        self.train_X = self.train_X/255.0
        self.test_X = self.test_X/255.0
        return self

    @staticmethod
    def __sigmoid(z):
        return 1.0/(1+np.exp(-z))

    def __propagate_once(self):
        assert len(self.train_X == 0) == 209
        Z = np.dot(self.w.T, self.train_X.T) + self.b
        assert Z.shape == (1, 209)
        A = self.__sigmoid(Z)
        assert A.shape == (1, 209)
        assert self.train_Y.shape == (1, 209)
        cost_array = np.dot(self.train_Y, np.log(A).T) + np.dot(1-self.train_Y, np.log(1-A).T)
        cost = -1.0/self.sample_amount * np.sum(cost_array)
        dw = (1.0/self.sample_amount) * np.dot(self.train_X.T, (A-self.train_Y).T)
        db = 1.0/self.sample_amount * np.sum((A - self.train_Y))
        assert db != 0
        return dw, db, cost

    def train(self, epoch=100, learning_rate=0.05, print_cost=True):
        self.is_trained = True
        for i in range(epoch):
            dw, db, cost = self.__propagate_once()
            self.w = self.w - learning_rate * dw
            self.b = self.b - learning_rate * db

            if print_cost and i % 100 == 0:
                print ("迭代 %i 次后: 损失函数均值为 %f" % (i, cost))
                # print(dw, db)

        f = h5py.File("iscat.h5", "w")
        f.create_dataset("w", data=self.w)
        f.create_dataset("b", data=self.b)
        print(self.w, self.b)

    def load_weight(self):
        self.w = np.array(h5py.File(self.weight_save_path, "r")["w"])
        self.b = np.array(h5py.File(self.weight_save_path, "r")["b"])
        self.is_trained = True
        return self

    def predict(self):
        assert self.is_trained
        self.faltten_x().standardize_x()
        Y_prediction = self.__sigmoid(np.dot(self.w.T, self.test_X.T) + self.b)
        assert Y_prediction.shape == (1, self.test_X.shape[0])
        print(self.test_Y)
        print(Y_prediction)
        Y_prediction[Y_prediction < 0.5] = 0
        Y_prediction[Y_prediction >= 0.5] = 1

        contrast_array = self.test_Y - Y_prediction
        print float(len(contrast_array[contrast_array==0]))/self.test_X.shape[0]
        return self

    def predict_image(self, image_path):
        image = np.array(ndimage.imread(image_path, flatten=False))
        my_image = scipy.misc.imresize(image, size=(64, 64)).reshape((1, 64 * 64 * 3))
        my_image = my_image/255.0
        self.load_weight()

        Y_prediction = self.__sigmoid(np.dot(self.w.T, my_image.T) + self.b)

        return Y_prediction

if __name__ == '__main__':
    # LogisticRegression().faltten_x().standardize_x().train(epoch=1500, learning_rate=0.005) # 17396.483413 s
    LogisticRegression().show_data_info().load_weight().predict()

    prob = np.squeeze(LogisticRegression().predict_image("images/ cat.jpg"))
    print prob
    print("这%s一只猫" % ("是" if prob> 0.5 else "不是"))
