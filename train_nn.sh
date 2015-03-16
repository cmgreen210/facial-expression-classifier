#!/bin/bash

# MODEL is the path to the neural net config
# MAX_IT is the maximum number of iterations
# DATA Path to pickled fer files
MODEL=$1
MAX_IT=$2
DATA=$3
EMAIL=$4

PY='/usr/bin/python'

if [ ! -d "checkpoint" ]; then
    mkdir "checkpoint"
fi

STAMP=$(date +"%Y%m%d_%H%M%S")
OUT_DIR="checkpoint/train_$STAMP"

if [ -d "${OUT_DIR}" ]; then
    rm -rf "${OUT_DIR}"
fi

mkdir "${OUT_DIR}"
mkdir "${OUT_DIR}/chkpt"


PARAMS="${OUT_DIR}/net.conf"
cp "${MODEL}" "${OUT_DIR}/net.conf"

OUTPUT="${OUT_DIR}/output.txt"

${PY} train_nn.py \
    "${PARAMS}" "${OUT_DIR}" "${DATA}" "${MAX_IT}"\
> "${OUTPUT}"

TAR_OUTPUT="${OUT_DIR}.tar.gz"

tar -zcvf "${TAR_OUTPUT}" "${OUT_DIR}"

cat "${OUT_DIR}/results.txt" | mutt -s "NN Training Run" -- "${EMAIL}"
aws s3 cp "${TAR_OUTPUT}" "s3://cmgreen210-emotions/"
