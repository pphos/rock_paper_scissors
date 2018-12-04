# -*- coding: utf-8 -*-
import os
from keras.utils import plot_model
from abc import ABCMeta, abstractmethod


class BaseModel(metaclass=ABCMeta):
    """
    識別モデルの抽象クラス
    """

    def __init__(self, name, input_shape, nb_classes,
                 save_dir, callbacks=None, **kwargs):
        self.name = name
        self.input_shape = input_shape
        self.nb_classes = nb_classes
        self.save_dir = save_dir

        self.history = None
        self.callbacks = callbacks

        self.loss = None
        self.optimizer = None
        self.metrics = None

    @classmethod
    @abstractmethod
    def define_model(cls):
        """
        self.nameに対応するモデルを構築する抽象メソッド
        """
        raise NotImplementedError()

    def plot_model_structure(self):
        """
        self.save_dirにモデル構造を描画した画像の保存
        """
        plot_name = '{}_model.png'.format(self.name)
        plot_path = os.path.join(self.save_dir, plot_name)
        plot_model(self.model, to_file=plot_path, show_shape=True)

    def summary(self):
        """
        モデルのサマリーの表示
        """
        return self.model.summary()

    def compile(self, loss, optimizer, metrics):
        """
        モデルのコンパイル
        """
        self.loss = loss
        self.optmizer = optimizer
        self.metrics = metrics

        self.model.compile(loss=loss,
                           optimizer=optimizer,
                           metrics=metrics)
