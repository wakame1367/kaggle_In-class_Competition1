import pandas as pd
import numpy as np
from scipy.misc import imresize
from base.base_data_loader import BaseDataLoader
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator


class ConvFashionMnistDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(ConvFashionMnistDataLoader, self).__init__(config)
        train_path = r"C:\Users\hotwa\PycharmProjects\kaggle_In-class_Competition1\data\train.csv"
        test_path = r"C:\Users\hotwa\PycharmProjects\kaggle_In-class_Competition1\data\test.csv"
        height, width = 56, 56
        train = pd.read_csv(train_path)
        self.X_test = pd.read_csv(test_path)

        self.X_train = train.drop(labels=["label"], axis=1)

        # images reshape
        self.X_train = self.X_train.values.reshape(-1, 28, 28)
        self.X_train = np.array([imresize(x, (height, width))
                                 for x in iter(self.X_train)])
        self.X_test = self.X_test.values.reshape(-1, 28, 28)
        self.X_test = np.array([imresize(x, (height, width))
                                for x in iter(self.X_test)])

        self.X_train = self.X_train.astype('float32')
        self.X_test = self.X_test.astype('float32')
        self.X_train /= 255
        self.X_test /= 255

        self.num_classes = 10
        # labels to category
        self.y_train = train["label"].values
        self.y_train = to_categorical(self.y_train, self.num_classes)

    def get_train_data(self):
        return self.X_train, self.y_train

    def get_test_data(self):
        return self.X_test


class ConvFashionMnistDataLoader2(BaseDataLoader):
    def __init__(self, config):
        super(ConvFashionMnistDataLoader2, self).__init__(config)
        train_path = r"C:\Users\hotwa\PycharmProjects\kaggle_In-class_Competition1\data\train.csv"
        test_path = r"C:\Users\hotwa\PycharmProjects\kaggle_In-class_Competition1\data\test.csv"
        height, width = 32, 32
        train = pd.read_csv(train_path)
        self.X_test = pd.read_csv(test_path)

        self.X_train = train.drop(labels=["label"], axis=1)

        # images reshape
        self.X_train = self.X_train.values.reshape(-1, 28, 28)
        self.X_train = np.array([imresize(x, (height, width))
                                 for x in iter(self.X_train)])
        self.X_test = self.X_test.values.reshape(-1, 28, 28)
        self.X_test = np.array([imresize(x, (height, width))
                                for x in iter(self.X_test)])

        self.X_train = self.X_train.astype('float32')
        self.X_test = self.X_test.astype('float32')
        self.X_train /= 255
        self.X_test /= 255

        self.num_classes = 10
        # labels to category
        self.y_train = train["label"].values
        self.y_train = to_categorical(self.y_train, self.num_classes)

    def get_train_data(self):
        return self.X_train, self.y_train

    def get_test_data(self):
        return self.X_test


class FashionMnistDataGenerator(BaseDataLoader):
    def __init__(self, config):
        super(FashionMnistDataGenerator, self).__init__(config)
        train_path = r"C:\Users\hotwa\PycharmProjects\kaggle_In-class_Competition1\data\train.csv"
        test_path = r"C:\Users\hotwa\PycharmProjects\kaggle_In-class_Competition1\data\test.csv"
        height, width = 56, 56
        train = pd.read_csv(train_path)
        self.X_test = pd.read_csv(test_path)

        self.X_train = train.drop(labels=["label"], axis=1)

        # images reshape
        self.X_train = self.X_train.values.reshape(-1, 28, 28, 1)
        self.X_train = np.array([imresize(x, (height, width))
                                 for x in iter(self.X_train)])
        self.X_test = self.X_test.values.reshape(-1, 28, 28, 1)
        self.X_test = np.array([imresize(x, (height, width))
                                for x in iter(self.X_test)])

        self.X_train = self.X_train.astype('float32')
        self.X_test = self.X_test.astype('float32')
        self.X_train /= 255
        self.X_test /= 255

        self.num_classes = 10
        # labels to category
        self.y_train = train["label"].values
        self.y_train = to_categorical(self.y_train, self.num_classes)
        self.X_train, self.y_train, self.X_val, self.y_val = train_test_split(self.X_train,
                                                                              self.y_train,
                                                                              test_size=0.1,
                                                                              random_state=43)
        self.train_generator = ImageDataGenerator(rotation_range=20,
                                                  width_shift_range=0.2,
                                                  height_shift_range=0.2,
                                                  horizontal_flip=True)
        self.val_generator = ImageDataGenerator(rotation_range=20,
                                                width_shift_range=0.2,
                                                height_shift_range=0.2,
                                                horizontal_flip=True)
        self.train_generator.fit(self.X_train)
        self.val_generator.fit(self.X_val)

    def get_train_data(self):
        return self.train_generator.flow(self.X_train, self.y_train),\
               self.val_generator.flow(self.X_val, self.y_val)

    def get_test_data(self):
        return self.X_test
