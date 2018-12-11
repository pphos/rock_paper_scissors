# -*- coding :utf-8 -*-
import argparse
from sklearn.utils import shuffle

from modules.utils import Bunch
from modules.utils.io import (
    load_dataset,
    init_result_save_dir
)
from modules.utils.preprocess import adjust_to_keras_input_image
from modules.models.model_selection import select_model


def load_and_preprocess_dataset(data_path, target_path,
                                target_label_path, random_state=12345):
    """
    データセットの読み込みと前処理の統合
    # Arguments
        data_path           : 入力データのパス (Numpy形式)
        target_path         : 教師データのパス (Numpy形式)
        target_label_path   : 教師ラベルのパス (pickle形式)
    # Returns
        preprocessed_dataset: Bunch オブジェクト
         Attribute)
            X               : 前処理後の入力データ (Numpy配列)
            y               : 前処理後の教師データ (Numpy配列)
            target_label    : 教師ラベル (pickle形式)
         ex)
            dataset = load_and_preprocess_dataset(data_path, target_path, target_label_path)
            X = dataset.X
            y = dataset.y
            target_label = dataset.target_label
    """
    # データセットの読み込み
    dataset = load_dataset(data_path, target_path, target_label_path)
    X = dataset.data
    y = dataset.target
    nb_classes = len(dataset.target_label)

    # データセットの前処理
    X, y = adjust_to_keras_input_image(X, y, nb_classes)
    X, y = shuffle(X, y, random_state=random_state)

    preprocessed_dataset = Bunch(X=X, y=y, target_label=dataset.target_label)

    return preprocessed_dataset


def init_each_config(args, target_label):
    """
    各種コンフィグの初期化
    # Arguments:
        args            : コマンドライン引数
        target_label    : 正解ラベル (pickle形式)
    # Returns
        model_conf      : モデル情報を格納した辞書 
        save_conf       : 結果の保存に関する情報を格納した辞書
    """
    # 保存対象ディレクトリの初期化
    model_name = args.model_name
    save_dirs = init_result_save_dir(name=model_name,
                                     save_root=args.save_root,
                                     unique=args.unique_dir)

    # model_conf の初期化
    model_conf = {
        'name': model_name,
        'save_dir': save_dirs['model'],
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'validation_split': args.validation_split,
        'loss': args.loss,
        'optimizer': args.optimizer,
        'metrics': [args.metrics],
        'set_class_weight': args.set_class_weight,
        'enable_early_stopping': args.enable_early_stopping
    }

    # save_confの初期化
    save_conf = {
        'save_dir': save_dirs['figure'],
        'digits': args.digits,
        'target_names': target_label,
        'cnf_matrix_title': args.cnf_matrix_title,
        'cnf_matrix_fname': args.cnf_matrix_fname,
        'text_fname': args.text_fname,
        'save_prefix': args.save_prefix
    }

    return model_conf, save_conf


def __get_arguments():
    """
    コマンドライン引数の取得
    """
    parser = argparse.ArgumentParser()

    # 必須引数の設定
    parser.add_argument(dest='model_name',
                        help="使用するモデル名")
    parser.add_argument(dest='save_root',
                        help='結果の保存ディレクトリのルートパス')

    parser.add_argument(dest='train_data',
                        help='訓練用の入力データのパス (Numpy形式)')
    parser.add_argument(dest='train_target',
                        help='訓練用の教師データのパス (Numpy形式)')
    parser.add_argument(dest='train_target_label',
                        help='訓練用の教師ラベルのパス (pickle形式)')

    parser.add_argument(dest='test_data',
                        help='評価用の入力データのパス (Numpy形式)')
    parser.add_argument(dest='test_target',
                        help='評価用の正解データのパス (Numpy形式)')
    parser.add_argument(dest='test_target_label',
                        help='評価用の正解データラベルのパス (pickle形式)')

    # 任意引数
    parser.add_argument('-ud', '--unique_dir', dest='unique_dir',
                        help='保存ディレクトリ名にユニークな名前を与えるかのフラグ',
                        action='store_true')

    # model_conf の引数設定
    model_args = parser.add_argument_group('model_conf arguments')

    model_args.add_argument('-e', '--epochs', dest='epochs',
                            help='モデルを訓練するエポック数 (int)',
                            default=100, type=int)
    model_args.add_argument('-bs', '--batch_size', dest='batch_size',
                            help='勾配の更新を行うバッチサイズ (int)',
                            default=32, type=int)
    model_args.add_argument('-vs', '--validation_split',
                            dest='validation_split',
                            help='訓練データの中で検証データとして使う割合 (float)',
                            default=0.1, type=float)
    model_args.add_argument('-l', '--loss', dest='loss',
                            help='損失関数',
                            default='categorical_crossentropy')
    model_args.add_argument('-o', '--optimizer', dest='optimizer',
                            help='最適化関数',
                            default='adam')
    model_args.add_argument('-m', '--metrics', dest='metrics',
                            help='評価やテストの際にモデルを評価するための評価関数のリスト',
                            default='accuracy')
    model_args.add_argument('-scw', '--set_class_weight',
                            dest='set_class_weight',
                            help='クラスごとに異なる重み付けを行う',
                            action='store_true')
    model_args.add_argument('-ee', '--enable_early_stopping',
                            dest='enable_early_stopping',
                            help='Early Stoppingにより学習の早期終了を行う')

    # save_conf の引数設定
    save_args = parser.add_argument_group('save_conf arguments')
    save_args.add_argument('-d', '--digits', dest='digits',
                           help='出力結果の有効数字の桁数',
                           default=4, type=int)
    save_args.add_argument('-cmt', '--cnf_matrix_title',
                           dest='cnf_matrix_title',
                           help='Confusion Matrixの図につけるタイトル',
                           default=None)
    save_args.add_argument('-cmf', '--cnf_matrix_fname',
                           dest='cnf_matrix_fname',
                           help='Confusion Matrixの保存名 (画像ファイル)',
                           default='confusion_matrix.png')
    save_args.add_argument('-tf', '--text_fname',
                           dest='text_fname',
                           help='classification_reportの保存名 (テキストファイル)',
                           default='classification_report.txt')
    save_args.add_argument('-sp', '--save_prefix',
                           dest='save_prefix',
                           help='各ファイルに付与するプレフィクス',
                           default=None)

    return parser.parse_args()


if __name__ == '__main__':
    # コマンドライン引数の取得
    args = __get_arguments()

    # データセットの読み込み兼前処理
    train_dataset = load_and_preprocess_dataset(args.train_data,
                                                args.train_target,
                                                args.train_target_label)
    test_dataset = load_and_preprocess_dataset(args.test_data,
                                               args.test_target,
                                               args.test_target_label)

    model_conf, save_conf = init_each_config(args, train_dataset.target_label)

    # 訓練モデルの選択
    model = select_model(model_conf['name'])
    model = model(name=model_conf['name'],
                  input_shape=train_dataset.X.shape[1:],
                  nb_classes=len(train_dataset.target_label),
                  save_dir=model_conf['save_dir'])

    # モデルの訓練
    model.train(train_dataset.X, train_dataset.y, model_conf)
    # モデルの評価
    model.evaluate(test_dataset.X, test_dataset.y)
    # モデルの評価結果の可視化
    model.visuaize_result(save_conf)
