# -*- coding: utf-8 -*-
import os
import numpy as np
from keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)
from sklearn.utils.class_weight import compute_class_weight


def train_model(X_train, y_train, model, model_conf):
    """
    モデルの学習
    # Arguments:
        X_train     : 訓練データ (Numpy配列)
        y_train     : 教師データのOne-hot表現 (Numpy配列)
        model       : Modelオブジェクト
        model_conf  : 学習に用いるパラメータを格納した辞書
    # Returns:
        history     : historyオブジェクト
    """
    # コールバック関数の設定
    callbacks = configure_callbacks(model_conf)

    # 指定がある場合にクラスごとに重み付けを行う
    if model_conf['set_class_weight']:
        class_weight_dict = calc_class_weight(y_train)
    else:
        # Keras のデフォルトではclass_weightはNone
        class_weight_dict = None

    # モデルのコンパイル
    model.compile(loss=model_conf['loss'],
                  optimizer=model_conf['optimizer'],
                  metrics=model_conf['metrics'])

    # モデルの学習
    history = model.fit(X_train, y_train,
                        batch_size=model_conf['batch_size'],
                        epochs=model_conf['epochs'],
                        verbose=1,
                        validation_split=model_conf['validation_split'],
                        callbacks=callbacks,
                        class_weight=class_weight_dict)

    # モデルの保存
    model_save_name = '{}_model.h5'.format(model_conf['name'])
    model_save_path = os.path.join(model_conf['save_dir'], model_save_name)
    model.save(model_save_path)

    return history


def calc_class_weight(y_train):
    """
    クラスごとに異なる重みの設定
    # Arguments:
        y   : 教師データのOne-hot表現
    # Returns:
        class_weight_dict : クラスごとの重みを格納した辞書:
    """
    y = y_train.argmax(axis=1)
    class_weight = compute_class_weight('balanced', np.unique(y), y)

    class_weight_dict = {}
    for index, weight in enumerate(class_weight):
        class_weight_dict[index] = weight

    return class_weight_dict


def configure_callbacks(model_conf):
    """
    コールバック関数の設定
    # Arguments:
        model_conf  : 学習に用いるパラメータを格納した辞書
    # Returns:
        callbacks   : コールバック関数を格納したリスト
    """
    # コールバック関数の設定
    callbacks = []
    if model_conf['enable_early_stopping']:
        # 監視する値の変化が停止したときに訓練を終了する
        callbacks.append(EarlyStopping(monitor='val_loss',
                                       min_delta=0.001,
                                       patience=15,
                                       verbose=1,
                                       mode='min'))
    # 評価値の改善が止まった時に学習率を減らす
    callbacks.append(ReduceLROnPlateau(factor=np.sqrt(0.1),
                                       cooldown=0,
                                       patience=5,
                                       min_lr=0.5e-6))

    # モデルの重みを保存
    save_weight_name = '{}_weights.{{epoch:03d}}-{{loss:.4f}}.hdf5'\
        .format(model_conf['name'])
    save_weight_path = os.path.join(model_conf['save_dir'], save_weight_name)
    callbacks.append(ModelCheckpoint(filepath=save_weight_path,
                                     monitor='val_loss',
                                     verbose=1,
                                     save_best_only=True,
                                     save_weights_only=True,
                                     mode='min'))

    return callbacks
