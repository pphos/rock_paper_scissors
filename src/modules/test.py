# -*- coding :utf-8 -*-
import os
import pickle
import numpy as np
from sklearn.utils import shuffle
from keras.models import load_model

from utils.io import load_dataset
from utils.preprocess import adjust_to_keras_input_image


def test_model(model_path, weight_path, X_test, y_test):
    """
    モデルの評価
    # Arguments:
        model_path  : モデルのパス
        weight_path : 学習済みの重みへパス
        X_test      : 評価用データ (Numpy配列)
        y_test      : 評価用データのラベルのOne-hot表現 (Numpy配列)
    """
    # モデルと学習済みの重みの読み込み
    model = load_model(model_path)
    model.load_weights(weight_path)

    # モデルの評価
    loss, accurary = model.evaluate(X_test, y_test)

    # モデルが予測したクラスの計算
    y_prod = model.predict(X_test, verbose=1)
    y_prod = y_prod.argmax(axis=1)
    y_true = np.argmax(y_test, axis=1)


if __name__ == '__main__':
    # 各種データパスの指定
    data_path = '../../datasets/eval_features/data.npy'
    target_path = '../../datasets/eval_features/target.npy'
    target_label_path = '../../datasets/eval_features/target_label.pkl'
    model_path = '../../results/MnistCNN_model.h5'
    weight_path = '../../results/MnistCNN_weights.001-10.4630.hdf5'

    # データセットの読み込み
    dataset = load_dataset(data_path, target_path, target_label_path)
    X = dataset.data
    y = dataset.target
    nb_classes = len(dataset.target_label)

    # データの前処理
    X, y = adjust_to_keras_input_image(X, y, nb_classes)
    X, y = shuffle(X, y, random_state=12345)

    # モデルの評価
    test_model(model_path, weight_path, X, y)
