#!/bin/bash

flags=""

while getopts l:b:f flag
do
    case "${flag}" in
        l) ll_file_path_name=${OPTARG};;
        b) target_bc_file_path_name=${OPTARG};;
        f) flags="$flags ${OPTARG} ";;
    esac
done

opt -load ~/HIPCC_LLVM_4.3.1/lib/libMDFID.so -mdfid ${flags} -enable-new-pm=0  ${ll_file_path_name} -o ${target_bc_file_path_name}
