#!/bin/bash

EXT=hip
CURPATH=$(pwd)
SCRIPT_PATH=/mnt/sdb1/home/mzw/workspace/test_space/llvm_test/useful_command


for file in $(ls ./)
do
    if [ "${file##*.}" = ${EXT} ]; then
        sh ${SCRIPT_PATH}/get_ll_witharg.sh -p ${CURPATH} -n ${file%%.*}
    fi
done