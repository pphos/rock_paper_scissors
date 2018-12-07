# -*- coding :utf-8 -*-
import numpy as np
from keras.models import load_model


def test_model(X_test, y_test, model=None, model_path=None, weight_path=None):
    """
    モデルの評価
    # Arguments:
        model_path  : モデルのパス
        weight_path : 学習済みの重みへパス
        X_test      : 評価用データ (Numpy配列)
        y_test      : 評価用データのラベルのOne-hot表現 (Numpy配列)
    # Returns:
        result_dict : 評価結果を格納した辞書
            'loss'      : 評価用データについてのloss値
            'accurary'  : 評価用データについての精度
            'y_true'    : 評価用データの正解ラベル
            'y_pred'    : 評価用データの予測ラベル
    """
    # モデルと学習済みの重みの読み込み
    if (model_path is not None) and (weight_path is not None):
        model = load_model(model_path)
        model.load_weights(weight_path)

    # モデルの評価
    loss, accurary = model.evaluate(X_test, y_test)

    # モデルが予測したクラスの計算
    y_pred = model.predict(X_test, verbose=1)
    y_pred = y_pred.argmax(axis=1)
    y_true = np.argmax(y_test, axis=1)

    result_dict = {
        'loss': loss,
        'accurary': accurary,
        'y_true': y_true,
        'y_pred': y_pred
    }

    return result_dict
