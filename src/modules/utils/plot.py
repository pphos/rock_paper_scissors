# -*- coding: utf-8 -*-
import itertools
import numpy as np
import matplotlib.pyplot as plt

from .io import determine_save_name_path


def plot_loss_acc_history(history, save_dir, save_name,
                          save_prefix=None, title=None):
    """
    lossとaccuraryのヒストリーの描画
    # Arguments:
        history     : モデルのヒストリー
        save_dir    : グラフを保存するディレクトリへのパス
        save_name   : グラフの保存名
        save_prefix : グラフの保存名に対するprefix
        title       : グラフにつけるタイトル
    """
    fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10, 2), sharex=True)

    fig.subplots_adjust(wspace=0.4)
    if title is not None:
        fig.suptitle(title)

    # accuracy historyの描画
    axR.plot(history['acc'], marker=None)
    axR.plot(history['val_acc'], marker=None)
    axR.set_xlabel('epoch')
    axR.set_ylabel('accuracy')
    axR.grid(True)
    axR.legend(['accuracy', 'val_acc'], loc='upper left')

    # loss historyの描画
    axL.plot(history['loss'], marker=None)
    axL.plot(history['val_loss'], marker=None)
    axL.set_xlabel('epoch')
    axL.set_ylabel('loss')
    axL.grid(True)
    axL.legend(['loss', 'val_loss'], loc='upper left')

    # グラフの保存
    figure_path = determine_save_name_path(save_dir, save_name, save_prefix)
    fig.savefig(figure_path)
    plt.close()


def plot_confusion_matrix(cm, target_names, save_dir, save_name,
                          save_prefix=None, title=None,
                          normalize=False, cmap=plt.cm.Blues):
    """
    sklearn confusion matrix example
    URL :     URL:    https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    Confusion Matrixの描画
    # Arguments:
        cm              : confusion matrix (Numpy配列)
        target_names    : ラベル名のリスト
        save_dir        : グラフを保存するディレクトリへのパス
        save_name       : グラフの保存ファイル名
        save_prefix     : 保存ファイル名に付与するプレフィックス
        title           : グラフに付与するタイトル
        normalize       : Confusion Matrixを正規化するかのフラグ (Bool)
        cmap            : グラフのカラーマップ
            ex) cmap=plt.cm.Blues
    """
    plt.figure(figsize=(9, 9))
    plt.rcParams['font.size'] = 24

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    if title is not None:
        plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)

    fmt = '.2f' if normalize else 'd'
    threshold = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment='center',
                 color='white' if cm[i, j] > threshold else 'black')

    plt.gcf().set_tight_layout(True)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    figure_path = determine_save_name_path(save_dir, save_name, save_prefix)

    plt.savefig(figure_path)
    plt.close()
