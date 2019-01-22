#!/bin/bash

#
#  パラメータの設定
#

# 必須引数の設定
MODEL_NAME="MnistCNN"
SAVE_ROOT='../results'

TRAIN_DATA_PATH="../splited_dataset/training_features/data.npy"
TRAIN_TARGET_PATH="../splited_dataset/training_features/target.npy"
TRAIN_TARGET_LABEL_PATH="../splited_dataset/training_features/target_label.pkl"

TEST_DATA_PATH='../splited_dataset/eval_features/data.npy'
TEST_TARGET_PATH='../splited_dataset/eval_features/target.npy'
TEST_TARGET_LABEL_PATH='../splited_dataset/eval_features/target_label.pkl'


# 任意引数の設定
UNIQUE_DIR_FLAG=false


# model_confの引数設定
EPOCHS=1
BATCH_SIZE=1
VALIDATION_SPLIT=0.1
LOSS='categorical_crossentropy'
OPTIMIZER='rmsprop'
METRICS='accuracy'
SET_CLASS_WEIGHT_FLAG=false
ENABLE_EARLY_STOPPING_FLAG=false

# save_confの引数設定
DIGITS=4
CMF_MATRIX_TITLE=''
CMF_MATRIX_FNAME='confusion_matrix.png'
TEXT_FNAME='classification_report.txt'
SAVE_PREFIX=''


if ${UNIQUE_DIR_FLAG} ; then
    UNIQUE_DIR_FLAG='--unique_dir'
else
    UNIQUE_DIR_FLAG=''
fi

if ${SET_CLASS_WEIGHT_FLAG} ; then
    SET_CLASS_WEIGHT_FLAG='--set_class_weight'
else
    SET_CLASS_WEIGHT_FLAG=''
fi

if ${ENABLE_EARLY_STOPPING_FLAG} ; then
    ENABLE_EARLY_STOPPING_FLAG='--enable_early_stopping'
else
    ENABLE_EARLY_STOPPING_FLAG=''
fi


#
# Pythonの引数の設定
#
TRAIN_PATH="${TRAIN_DATA_PATH} ${TRAIN_TARGET_PATH} ${TRAIN_TARGET_LABEL_PATH}"
TEST_PATH="${TEST_DATA_PATH} ${TEST_TARGET_PATH} ${TEST_TARGET_LABEL_PATH}"

POSITIONAL_ARGS="${MODEL_NAME} ${SAVE_ROOT} ${TRAIN_PATH} ${TEST_PATH}"
OPTIONAL_ARGS="${UNIQUE_DIR_FLAG}"

MODEL_CONF="-e ${EPOCHS} -bs ${BATCH_SIZE} -vs ${VALIDATION_SPLIT} -l ${LOSS}\
            -o ${OPTIMIZER} -m ${METRICS} ${ENABLE_EARLY_STOPPING_FLAG} ${SET_CLASS_WEIGHT_FLAG}"

SAVE_CONF="-d ${DIGITS}"
if [ -n "$CMF_MATRIX_TITLE" ]; then
    SAVE_CONF="${SAVE_CONF} -cmt ${CMF_MATRIX_TITLE}"
fi

if [ -n "$SAVE_PREFIX" ]; then
    SAVE_PREFIX="${SAVE_CONF} -sp ${SAVE_PREFIX}"
fi


#
# Pythonスクリプトの実行
#
ARGUMENTS="${POSITIONAL_ARGS} ${OPTIONAL_ARGS}\
           ${MODEL_CONF} ${SAVE_CONF}"

RUN_COMMAND="python3 ./rock_paper_scissors.py ${ARGUMENTS}"

echo "========================================="
echo "${RUN_COMMAND}"
echo "========================================="

${RUN_COMMAND}
