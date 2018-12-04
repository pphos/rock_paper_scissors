# -*- coding: utf-8 -*-
import os
import pickle
import argparse
import numpy as np
import cv2


TRINING_FEATURE_NAME = 'training_features'
EVALUATION_FEATURE_NAME = 'eval_features'
GRAY_SCALE = 1


def convert_imgs_to_arrays(input_dir, dataset_type, save_dir, img_shape):
    """
    データセットの画像をNumpy配列に変換
    # Arguments
        input_dir   : 入力画像があるフォルダへのパス
        save_dir    : Numpy形式の画像を保存するフォルダへのパス
    """
    # 保存ファイル名の設定
    save_base_dir = os.path.join(save_dir, dataset_type)
    data_path = os.path.join(save_base_dir, 'data.npy')
    target_path = os.path.join(save_base_dir, 'target.npy')
    target_label_path = os.path.join(save_base_dir, 'target_label.pkl')
    if not os.path.exists(save_base_dir):
        os.makedirs(save_base_dir)

    # カテゴリの取得
    categories = os.listdir(input_dir)
    # データを保存するNumpy配列の初期化
    data, target = __init_np_array(input_dir, categories, img_shape)

    progress = 0
    for index, category in enumerate(categories):
        category_path = os.path.join(input_dir, category)
        for image_name in os.listdir(category_path):
            image_path = os.path.join(category_path, image_name)

            # 画像のリサイズ
            image = resize_image(image_path, img_shape)
            # Numpy配列に訓練・教師データの格納
            data[progress] = image
            target[progress] = index

            progress += 1
            print("{:04d}: {} {:10s} :{}".format(progress, index,
                                                 category, image_name))

    # 訓練・教師データおよびラベルの保存
    with open(target_label_path, 'wb') as f:
        pickle.dump(categories, f)
    np.save(data_path, data[:progress])
    np.save(target_path, target[:progress])


def resize_image(img_path, img_shape):
    """
    画像のリサイズ
    # Arguments
        img_path    : 画像のパス
        img_shape   : リサイズする画像のタプル
    # Returns
        img_resized : リサイズ後の画像のNumpy配列
    """
    image = cv2.imread(img_path)
    img_resized = cv2.resize(image, (img_shape, img_shape))

    return img_resized


def __init_np_array(input_dir, categories, img_shape):
    """
    データを保存するNumpy配列の初期化
    # Arguments:
        input_dir   : 入力画像があるフォルダへのパス
        categories  : 画像のカテゴリを格納したリスト
        img_shape   : リサイズする画像サイズ
    # Returns:
        data        : 全要素が0の訓練データ用のNumpy配列
        target      : 全要素が0の教師データ用のNumpy配列
    """
    # データセット内の総ファイル数の算出
    total_file_nums = __calc_total_file_nums(input_dir, categories)
    target = np.zeros((total_file_nums,), dtype='int32')

    # 1ファイル分の画像を読み込み,
    # total_file_numsを元に必要なNumpy配列のサイズを算出
    data = None
    for root, dirs, files in os.walk(input_dir):
        for fname in files:
            # 1ファイル分の画像のリサイズ
            img_path = os.path.join(root, fname)
            img = resize_image(img_path, img_shape)

            if len(img.shape) == GRAY_SCALE:
                data = np.zeros((total_file_nums, img_shape, img_shape))
            else:
                data = np.zeros((total_file_nums,
                                 img_shape, img_shape, img.shape[-1]))
            break

        if data is not None:
            break

    return data, target


def __calc_total_file_nums(input_dir, categories):
    """
    データセット内の総ファイル数の算出
    # Arguments:
        input_dir       : 入力画像があるフォルダへのパス
        categories      : 画像のカテゴリを格納したリスト
    # Returnes
        total_file_nums : データセット内の総ファイル数
    """
    total_file_nums = 0
    for category in categories:
        category_path = os.path.join(input_dir, category)
        total_file_nums += len(os.listdir(category_path))

    return total_file_nums


def __get_arguments():
    """
    コマンドライン引数の取得
    # Arguments:
        args    : Namespaceオブジェクト
    """
    parser = argparse.ArgumentParser()

    # 必須引数の設定
    parser.add_argument(dest="input_dir",
                        help="入力画像があるフォルダへのパス")
    parser.add_argument(dest="save_dir",
                        help="Numpy形式の画像を保存するフォルダへのパス")

    # 任意引数の設定
    parser.add_argument('-is', '--img_shape', dest='img_shape',
                        help="リサイズする画像のサイズ",
                        default=256, type=int)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # コマンドライン引数の取得
    args = __get_arguments()

    print("=== START Convert Images to Ndarrays ====")

    # 訓練セットをNdarraysに変換
    convert_imgs_to_arrays(args.input_dir, TRINING_FEATURE_NAME,
                           args.save_dir, args.img_shape)
    # 評価セットをNdarraysに変換
    convert_imgs_to_arrays(args.input_dir, EVALUATION_FEATURE_NAME,
                           args.save_dir, args.img_shape)

    print("==== FINISH Convert Image to Ndarrays =====")
