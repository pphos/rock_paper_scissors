# -*- coding: utf-8 -*-
def select_model(name):
    """
    nameに対応するモデルのクラスを返却する
    """
    if name == 'MnistCNN':
        from .mnist_cnn import MnistCNN
        return MnistCNN

    else:
        raise NotImplementedError()
