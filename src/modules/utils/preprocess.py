# -*- coding: utf-8 -*-
from sklearn.utils import shuffle
from keras.utils import to_categorical
from . import Bunch
from .io import load_dataset


def load_and_preprocess_dataset(data_path, target_path,
                                target_label_path, random_state=12345):
    """
    データセットの読み込みと前処理の統合
    # Arguments
        data_path           : 入力データのパス (Numpy形式)
        target_path         : 教師データのパス (Numpy形式)
        target_label_path   : 教師ラベルのパス (pickle形式)
    # Returns
        preprocessed_dataset: Bunch オブジェクト
         Attribute)
            X               : 前処理後の入力データ (Numpy配列)
            y               : 前処理後の教師データ (Numpy配列)
            target_label    : 教師ラベル (pickle形式)
         ex)
            dataset = load_and_preprocess_dataset(data_path, target_path, target_label_path)
            X = dataset.X
            y = dataset.y
            target_label = dataset.target_label
    """
    # データセットの読み込み
    dataset = load_dataset(data_path, target_path, target_label_path)
    X = dataset.data
    y = dataset.target
    nb_classes = len(dataset.target_label)

    # データセットの前処理
    X, y = adjust_to_keras_input_image(X, y, nb_classes)
    X, y = shuffle(X, y, random_state=random_state)

    preprocessed_dataset = Bunch(X=X, y=y, target_label=dataset.target_label)

    return preprocessed_dataset


def adjust_to_keras_input_image(X, y, nb_classes):
    """
    画像データをKerasの入力形式合わせる
    # Arguments:
    X         : 訓練画像のNumpy配列
    y         : 教師ラベルのNumpy配列
    nb_classes: 分類すべきクラス数

    # Returns:
    X : Kerasの入力形式に合わせた訓練画像のNumpy配列
    y : 教師ラベルのOne-hot表現
    """

    # 訓練画像サイズの取得
    img_x = X.shape[1]
    img_y = X.shape[2]

    if len(X.shape) == 4:
        num_channels = X.shape[-1]
    else:
        num_channels = 1

    X = X.reshape(len(X), img_x, img_y, num_channels)
    # 画像データの各画素を[0, 1]に正規化する
    X = X.astype('float32') / 255.0
    # 教師ラベルをOne-hot表現に変換
    y = to_categorical(y, nb_classes)

    return X, y
