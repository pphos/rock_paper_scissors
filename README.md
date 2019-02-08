# Rock Paper Scissors

Kerasによるじゃんけん画像の分類。
データセットを自前で用意すれば、他の画像分類も可能。

画像の評価後、モデルアーキテクチャ, Tensorboardを用いるためのログ,混同行列の表, 適合率・再現率・F値・サンプル数を示したテキストが保存される。

## Requirement
Python3.6
graphviz (訓練結果の描画用)

## Installation
```bash
$ git clone https://github.com/rock_paper_scissors.git
$ cd ./rock_paper_scissors.git
$ pipenv shell
$ pipenv install
```
pipenvを用いない場合は、Pipfile内[packages]のライブラリを手動でインストールする。

## Usage 
### ・リポジトリ内のデータセットをそのまま用いる場合
### データセットの解凍
    $ cat ./datasets/rock_paper_scissors.tar.gz.* > datasets.tar.gz
    $ tar -zxvf datasets.tar.gz
    $ rm -rf datasets.tar.gz datasets
    
### 画像分類の実行	
```$ ./run.sh```
スクリプトをそのまま実行すると結果がrock_paper_scissors/results以下に格納される。


### パラメータの説明
#### 必須パラメータ
```MODEL_NAME``` : 使用するモデル名
 - 現在、'MnistCNN'モデルしかない
 
```SAVE_ROOT``` : 結果を保存するディレクトリのパス

```TRAIN_DATA_PATH``` : 訓練用データのパス (Numpy形式の画素値)  
```TRAIN_TARGET_PATH``` : 訓練用の教師データのパス (Numpy形式)  
```TRAIN_TARGET_LABEL_PATH``` : 訓練用の教師ラベルのパス (pickle形式)   

```TEST_DATA_PATH``` : 評価用の入力データのパス (Numpy形式の画素値)  
```TEST_TARGET_PATH```: 評価用の正解データのパス (Numpy形式)   
```TEST_TARGET_LABEL_PATH``` : 評価用の正解ラベルのパス (pickle形式)   
#### 任意パラメータ
```UNIQUE_DIR_FLAG``` : 結果を保存するディレクトリにユニーク名前を与えるかのフラグ ( bool )  

#### モデルに関するパラメータ
```EPOCHS``` : モデルを訓練するエポック数 ( int )  
```BATCH_SIZE``` : 勾配の更新を行うバッチサイズ (int)  
```VALIDATION_SPLIT``` :    訓練データの中で検証用データとして使う割合 ( float )  
```loss``` : 損失関数  
- kerasドキュメント[損失関数](https://keras.io/ja/losses/)にある損失関数が指定可能  

```optimizer```: 最適化関数  
- kerasドキュメント[最適化](https://keras.io/ja/optimizers/)にある最適化関数が指定可能  

```SET_CLASS_WEIGHT_FLAG``` : クラスごとに重み付けをするかのフラグ ( bool )  
```ENABLE_EARLY_STOPPING_FLAG``` : Early Stoppingにより学習の早期終了を行うかのフラグ ( bool )  

#### 結果の出力に関するパラメータ
```DIGITS``` : 出力結果の有効数字の桁数 ( int )  
```CMF_MATRIX_TITLE``` : Confusion Matrixの図につけるタイトル  
```CMF_MATRIX_FNAME``` : Confustion Matrixの保存名 ( 画像ファイル )   
```TEXT_FNAME``` : classification_reportの保存名 (テキストファイル )   
```SAVE_PREFIX``` : 各ファイルに付与するプレフィクス

### ・データセットを自分で作成する場合
- rock_paper_scissors/src/preprocess以下のファイルを使用する
	- capture_image.py : Webカメラから読み込んだ画像の保存を行う
	- build_dataset.py : 生のデータセットを訓練・評価セットに分割する
	- convert_img_to_array.py: データセットの画像をNumpy配列に変換する

1. 画像のキャプチャ
	```
	$ python3 capture_image.py 0 'rock' ../../datasets/raw -n 1000
	$ python3 capture_image.py 0 'paper' ../../datasets/raw -n 1000
	$ python3 capture_image.py 0 'scissors' ../../datasets/raw -n 1000
	```
	#### パラメータの説明
	```$ python3 capture_image.py <device_code> <label> <save_root_path> <sampling_times>```
	
	```device_code``` : Webカメラのデバイスコード (カメラが一台しかなければ 0 )  
	```label``` : 画像に付与するラベル  
	```save_root_path``` :  画像を保存するディレクトリへのパス  
	```-n``` , ```sampling_times``` : Webカメラから保存する画像の枚数  
	

2. データセットを訓練用と評価用に分割
	```$ python3 build_dataset.py ../../datasets/raw ../../datasets```
	#### パラメータの説明
	``` $ python3 build_dataset.py <raw_data_dir> <save_dir> ```

	```raw_data_dir``` : キャプチャした画像セットのディレクトリパス  
	```save_dir``` : 評価・訓練データの保存先  

3. 訓練・評価セットの画像をそれぞれNumpy配列に変換  
	```$ python3 convert_imgs_to_arrays.py ../../datasets/training_set ../../datasets/training_features```
	```$ python3 convert_imgs_to_arrays.py ../../datasets/eval_set ../../datasets/eval_features```

4. スクリプトにdatasetのパスを書き入れて画像分類の実行  
```$ ./run.sh```
git clone直後は```run.sh```において学習に用いるデータセットの起点パスが```DATASET_BASE_DIR='../splited_dataset```となっているので、この部分を各々に合うように修正する。
3.の例では、```DATASET_BASE_DIR='../datasets'```となる。

### ・Google Colaboratoryを使用する場合
```rock_paper_scissors.ipynb```を実行するだけ。
モデルのパラメータ調整は```init_each_config```内の値を変更することで可能。
