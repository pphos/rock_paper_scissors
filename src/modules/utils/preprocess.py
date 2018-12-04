# -*- coding: utf-8 -*-
from keras.utils import to_categorical


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


def resize_images(img_path, img_shape):
    """
    画像のリサイズ
    # Arguments
    img_path    : 画像のパス
    img_shape   : リサイズする画像のサイズのタプル
                  (ex. img_shape = (img_x, img_y))

    # Returns:
    img_resized : リサイズ後の画像のNumpy配列
    """
    import cv2

    image = cv2.imread(img_path)
    img_resized = cv2.resize(image, img_shape)

    return img_resized


if __name__ == '__main__':
    img_path = '../../../tiny_datasets/内田真礼/30263767.jpeg'

    print(resize_images(img_path, (256, 256)))