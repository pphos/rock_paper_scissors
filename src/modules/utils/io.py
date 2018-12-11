# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np

from . import Bunch


def determine_save_name_path(save_dir, save_name, save_prefix=None):
    """
    保存ファイル名の決定
    # Arguments:
        save_dir    : 保存先のディレクトリパス
        save_name   : 保存ファイル名
        save_prefix : ファイル名に付与するプレフィクス
    """

    if save_prefix is not None:
        file_name = '{}_{}'.format(save_prefix, save_name)
    else:
        file_name = save_name
    file_path = os.path.join(save_dir, file_name)

    return file_path


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


def write_message(message, filepath):
    """
    テキストデータの書き込み
    # Arguments:
        message     : 書き込み対象のメッセージ
        filepath    : 保存対象ファイル (txt形式)
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(message)


def load_dataset(data_path, target_path, target_label_path=None):
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

    if target_label_path is not None:
        target_label = load_binary_file(target_label_path)
        dataset = Bunch(data=data,
                        target=target,
                        target_label=target_label)
    else:
        dataset = Bunch(data=data, target=target)

    return dataset
