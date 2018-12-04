# -*- coding: utf-8 -*-
import os
import argparse
import cv2


ESC = 27


def capture_image(device_code, save_dir, sampling_times=1000):
    """
    Webカメラから読み込んだ画像の保存
    # Arguments
        device_code     : 用いるWebカメラのデバイスコード
        save_dir        : 画像の保存ディレクトリへのパス
        samplint_times  : 画像の保存枚数の指定 
    """

    label = os.path.basename(save_dir)

    # 使用するWebカメラの指定
    capture = cv2.VideoCapture(device_code)

    # フレームサイズの決定
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # フレームレートの決定
    capture.set(cv2.CAP_PROP_FPS, 30)

    if capture.isOpened() is False:
        raise("Device IO Error")

    for sampling_time in range(sampling_times):
        # Webカメラからの画像の取得
        ret, frame = capture.read()

        # 画像取得失敗時は再取得
        if ret is False:
            continue

        # フレームの表示
        cv2.imshow("Image Capture", frame)

        # 同名ファイルが存在している場合はファイル名に別番号を付与
        save_index = 0
        while True:
            if save_index == 0:
                frame_name = "{}_{}.png".format(label, sampling_time)
                frame_path = os.path.join(save_dir, frame_name)

            if os.path.exists(frame_path):
                save_index += 1
                frame_name = "{}_{}_{}.png"\
                    .format(label, sampling_time, save_index)
                frame_path = os.path.join(save_dir, frame_name)
            else:
                break

        # 画像の保存
        cv2.imwrite(frame_path, frame)
        print("{:04d} : {}".format(sampling_time, frame_name))

        # ESCが押された場合は規定枚数未満でも終了
        if cv2.waitKey(20) == ESC:
            break

    capture.release()
    cv2.destroyAllWindows()


def __get_arguments():
    """
    コマンドライン引数の取得
    # Returns:
        args : Namespaceオブジェクト
    """
    parser = argparse.ArgumentParser()

    # 必須引数の設定
    parser.add_argument(dest="device_code",
                        help="Webカメラのデバイスコード", type=int)
    parser.add_argument(dest="label",
                        help="画像に付与するラベル")
    parser.add_argument(dest="save_root_path",
                        help="画像を保存するディレクトリへのパス")

    # 任意引数の設定
    parser.add_argument('-n', dest='sampling_times',
                        help="Webカメラから保存する画像の枚数",
                        type=int, default=1000)

    return parser.parse_args()


if __name__ == '__main__':
    # コマンド引数の取得
    args = __get_arguments()

    # 保存先のディレクトリが存在しない場合に作成する
    save_dir_path = os.path.join(args.save_root_path, args.label)
    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)

    print("===== Start Capture Image ====")
    capture_image(args.device_code, save_dir_path, args.sampling_times)
    print("===== Fisnish Capture Image ====")
