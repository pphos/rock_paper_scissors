# -*- coding: utf-8 -*-
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight
from abc import ABCMeta, abstractmethod

from .callbacks import ModelCheckpoint
from ..utils.io import (
    determine_save_name_path,
    write_message
)
from ..utils.plot import (
    plot_confusion_matrix,
    plot_loss_acc_history
)


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

        self.callbacks = callbacks
        self.model = None
        self.history = None
        self.result_dict = None

    @classmethod
    @abstractmethod
    def struct_model(cls):
        """
        self.nameに対応するモデルを構築する抽象メソッド
        """
        raise NotImplementedError()

    def plot_model_structure(self):
        """
        self.save_dirにモデルの構造を描画
        """
        plot_path = os.path.join(self.save_dir,
                                 '{}_model.png'.format(self.name))
        plot_model(self.model, to_file=plot_path, show_shapes=True)

    def load(self, model_path, weight_path):
        """
        モデルと学習済み重みの読み込み
        # Arguments:
            model_path  : モデルのパス
            weight_path : 学習済み重みの読み込み
        """
        self.model = load_model(model_path)
        self.model.load_weights(weight_path)

    def train(self, X_train, y_train, model_conf, use_tpu=False):
        """
        モデルの学習
        # Arguments:
            X_train     : 訓練データ (Numpy配列)
            y_train     : 教師データのOne-hot表現 (Numpy配列)
            model_conf  : 学習に用いるパラメータを格納した辞書
            use_tpu     : TPUの利用設定 (Google Colaboratory用)
        # Returns:
            history     : historyオブジェクト
        """
        # コールバック関数の設定
        callbacks = _configure_callbacks(model_conf, use_tpu)

        # 指定がある場合にクラスごとに重み付けを行う
        if model_conf['set_class_weight']:
            class_weight_dict = _calc_class_weight(y_train)
        else:
            class_weight_dict = None

        # TPUの利用設定
        if use_tpu:
            from tensorflow.contrib.tpu import (
                keras_to_tpu_model,
                TPUDistributionStrategy,
            )
            from tensorflow.contrib.cluster_resolver import (
                TPUClusterResolver
            )
            model = keras_to_tpu_model(
                self.model,
                strategy=TPUDistributionStrategy(
                    TPUClusterResolver(
                        tpu='grpc://' + os.environ['COLAB_TPU_ADDR']
                    )
                )
            )
            model.compile(
                optimizer=model_conf['optimizer'],
                loss=model_conf['loss'],
                metrics=model_conf['metrics'])

        else:
            # モデルのコンパイル
            model = self.model
            model.compile(loss=model_conf['loss'],
                          optimizer=model_conf['optimizer'],
                          metrics=model_conf['metrics'])

        # モデルの学習
        history = model.fit(X_train, y_train,
                            batch_size=model_conf['batch_size'],
                            epochs=model_conf['epochs'],
                            validation_split=model_conf['validation_split'],
                            class_weight=class_weight_dict)

        # 訓練後TPUモデルをCPUモデルに変換
        if use_tpu:
            model = model.sync_to_cpu()

        # モデルの保存
        model_save_name = '{}_model.h5'.format(model_conf['name'])
        model_save_path = os.path.join(model_conf['save_dir'], model_save_name)
        model.save(model_save_path)

        # 訓練loss, accの描画
        save_name = "{}_training_loss_and_accuracy.png"\
            .format(model_conf['name'])
        plot_loss_acc_history(history.history,
                              model_conf['save_dir'], save_name)

        return history

    def evaluate(self, X_test, y_test):
        """
        モデルの評価
        # Arguments:
            X_test      : 評価用データ (Numpy配列)
            y_test      : 評価用データのラベルのOne-hot表現 (Numpy配列)
        # Returns:
            result_dict : 評価結果を格納した辞書
                'loss'      : 評価用データについてのloss値
                'accuracy'  : 評価用データについてのaccuracy
                'y_true'    : 評価用データの正解ラベル
                'y_pred'    : 評価用データの予測ラベル
        """
        # モデルの評価
        loss, accuracy = self.model.evaluate(X_test, y_test)

        # モデルが予測したクラスの計算
        y_pred = self.model.predict(X_test, verbose=1)
        y_pred = y_pred.argmax(axis=1)
        y_true = np.argmax(y_test, axis=1)

        result_dict = {
            'loss': loss,
            'accuracy': accuracy,
            'y_true': y_true,
            'y_pred': y_pred
        }
        self.result_dict = result_dict

        return result_dict

    def visuaize_result(self, save_conf, result_dict=None):
        """
        モデルの評価結果の可視化
        # Aruguments:
            save_conf   : 可視化結果の保存に関するパラメータを格納した辞書
            result_dict : モデルの評価結果を格納した辞書
        """
        if result_dict is not None:
            self.result_dict = result_dict

        # Confusion Matrixの算出
        cnf_matrix = confusion_matrix(self.result_dict['y_true'],
                                      self.result_dict['y_pred'])

        # 精度, 再現率, F値の算出
        clf_report = classification_report(self.result_dict['y_true'],
                                           self.result_dict['y_pred'],
                                           target_names=save_conf['target_names'],
                                           digits=save_conf['digits'])

        # loss, acc, clf_reportをテキストファイルで保存
        evaluate_message = '{}\n\n'.format(clf_report)
        evaluate_message += 'loss:  {}\n'.format(self.result_dict['loss'])
        evaluate_message += 'acc :  {}'.format(self.result_dict['accuracy'])

        text_path = determine_save_name_path(save_conf['save_dir'],
                                             save_conf['text_fname'],
                                             save_conf['save_prefix'])
        write_message(evaluate_message, text_path)

        # Confusion Matrixの描画
        save_prefix = save_conf['save_prefix']
        for normalize in [False, True]:
            if normalize:
                if save_prefix is not None:
                    save_prefix = '{}_{}'\
                        .format('normalize', save_conf['save_prefix'])
                else:
                    save_prefix = 'normalize'

            plot_confusion_matrix(cm=cnf_matrix,
                                  target_names=save_conf['target_names'],
                                  save_dir=save_conf['save_dir'],
                                  title=save_conf['cnf_matrix_title'],
                                  save_name=save_conf['cnf_matrix_fname'],
                                  save_prefix=save_prefix,
                                  normalize=normalize)


def _calc_class_weight(y_train):
    """
    クラスごとに異なる重みの設定
    # Arguments:
        y_train : 教師データOne-hot表現 (Numpy配列)
    # Returns:
        class_weight_dict : クラスごとの重みを格納した辞書
    """
    y = y_train.argmax(axis=1)
    class_weight = compute_class_weight('balanced', np.unique(y), y)

    class_weight_dict = {}
    for index, weight in enumerate(class_weight):
        class_weight_dict[index] = weight

    return class_weight_dict


def _configure_callbacks(model_conf, use_tpu=False):
    """
    コールバック関数の設定
    # Arguments:
        model_conf  : 学習に用いるパラメータを格納した辞書
        use_tpu     : TPUの使用フラグ
    # Returns:
        callbacks   : コールバック関数を格納したリスト
    """
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
    if not use_tpu:
        save_weight_name = '{}_weights.{{epoch:03d}}-{{loss:.4f}}.hdf5'\
            .format(model_conf['name'])
        save_weight_path =\
            os.path.join(model_conf['save_dir'], save_weight_name)
        callbacks.append(ModelCheckpoint(filepath=save_weight_path,
                                         monitor='val_loss',
                                         verbose=1,
                                         save_best_only=True,
                                         save_weights_only=True,
                                         mode='min'))

    # TensorBoardの利用設定
    log_dir = os.path.join(model_conf['save_dir'], 'log_dir')
    callbacks.append(TensorBoard(log_dir=log_dir,
                                 histogram_freq=1,
                                 write_grads=True,
                                 write_images=1))

    return callbacks
