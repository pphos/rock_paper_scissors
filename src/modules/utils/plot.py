# -*- coding: utf-8 -*-
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
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
