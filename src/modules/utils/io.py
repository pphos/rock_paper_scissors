# -*- coding: utf-8 -*-
import pickle
import numpy as np

from . import Bunch


def load_binary_file(filepath):
    """
    pickle形式のバイナリデータの読み込み
    # Arguments:
        filepath    : 読み込む対象ファイル (pickle形式)
    # Returns
        data        : 読み込んだpickleデータ
    """
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data


def save_binary_data(data, filepath):
    """
    pickle形式のバイナリデータを書き込み
    # Arguments:
        filepath    : 保存対象ファイル (pickle形式)
    """
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


def load_dataset(data_path, target_path, target_label_path):
    """
    データセットの読み込み
    # Arguments:
        data_path           : 訓練データのパス (Numpy形式)
        target_path         : 教師データのパス (Numpy形式)
        target_label_path   : データセットのラベル (pickle形式)
    # Returns:
        dataset             : Bunchオブジェクト
    ex)
        dataset = load(data_path, target_path, target_label_path)
        data = dataset.data
        target = dataset.target
        target_label = dataset.target_label
    """
    # 各種データの読み込み
    data = np.load(data_path)
    target = np.load(target_path)
    target_label = load_binary_file(target_label_path)

    dataset = Bunch(data=data, target=target,
                    target_label=target_label)

    return dataset
