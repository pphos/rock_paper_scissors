# -*- coding: utf-8 -*-
import os
import random
import shutil
import argparse


TRAINING_SET_NAME = "training_set"
EVALUATION_SET_NAME = "eval_set"
EVAL_RATIO = 0.25


def separate_train_test_data(raw_data_dir, save_dir):
    """
    生のデータセットを訓練セットと評価セットに分割
    # Arguments:
        raw_data_dir: 生のデータセット
                      階層構造: raw->category->image_path
        save_dir    : データを保存するディレクトリへのパス
    """
    # カテゴリの一覧を取得
    categories = os.listdir(raw_data_dir)
    for category in categories:
        category_path = os.path.join(raw_data_dir, category)
        images = os.listdir(category_path)

        # 評価データセットの大きさを決定
        eval_samples = int(len(images) * EVAL_RATIO)
        # データセットからランダムに評価セットを取得
        eval_sets = random.sample(images, eval_samples)
        # 残りデータを訓練セットとして取得
        train_sets = list(set(images) ^ set(eval_sets))

        # 生データから評価セットと訓練セットをコピー
        copy_images(eval_sets, EVALUATION_SET_NAME, category_path, save_dir)
        copy_images(train_sets, TRAINING_SET_NAME, category_path, save_dir)


def copy_images(images, dataset_type, category_path, save_dir):
    """
    datasetで指定したデータをsave_dirで指定した場所へコピーする
    # Arguments:
        dataset         : ファイル名のリスト
        dataset_type    : データセットの種別
                          (TRAINING_SET_NAME or EVALUATION_SET_NAME)
        category_path   : 画像が保存されているカテゴリへのパス
        save_dir        : 保存先のディレクトリ名
    """
    # 画像の保存先ディレクトリの作成
    category = os.path.basename(category_path)
    save_category_dir = os.path.join(save_dir, dataset_type, category)
    if not os.path.exists(save_category_dir):
        os.makedirs(save_category_dir)

    for progress, image in enumerate(images):
        input_image_path = os.path.join(category_path, image)
        print("{:10s} {}: {} -> {}"
              .format(category, progress, input_image_path, save_category_dir))
        shutil.copy(input_image_path, save_category_dir)


def __get_arguments():
    """
    コマンドライン引数の取得
    # Arguments
        args    : Namespaceオブジェクト
    """
    parser = argparse.ArgumentParser()

    # 必須引数の設定
    parser.add_argument(dest="raw_data_dir",
                        help="画像データセットのディレクトリへのパス")
    parser.add_argument(dest="save_dir",
                        help="評価・訓練データの保存先")

    return parser.parse_args()


if __name__ == '__main__':
    # コマンドライン引数の取得
    args = __get_arguments()

    print("==== START SPLIT DATASET ====")
    separate_train_test_data(args.raw_data_dir, args.save_dir)
    print("==== FINISH SPLIT DATASET =====")
