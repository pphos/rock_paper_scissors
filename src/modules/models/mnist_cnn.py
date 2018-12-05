# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

from .base_model import BaseModel


class MnistCNN(BaseModel):
    """
    Keras example Mnist CNN models
    URL: https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
    """

    def __init__(self, name, input_shape, nb_classes,
                 save_dir, **kwargs):
        super().__init__(name, input_shape, nb_classes,
                         save_dir, **kwargs)
        self.model = self.struct_model()
        self.plot_model_structure()

    def struct_model(self):
        """
        MnistCNN モデルの構築
        # Returns:
            model   : Modelオブジェクト
        """
        model = Sequential()
        model.add(Conv2D(8, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=self.input_shape))
        model.add(Conv2D(16, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.nb_classes, activation='softmax'))

        return model
