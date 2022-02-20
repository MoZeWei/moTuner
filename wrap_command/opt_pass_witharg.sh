#!/bin/bash

flags=""

while getopts n:s:l:t:f: flag
do
    case "${flag}" in
        n) pass_name=${OPTARG};;
        s) pass_so=${OPTARG};;
        l) ll_file_path_name=${OPTARG};;
        t) target_bc_file_path_name=${OPTARG};;
        f) flags="$flags ${OPTARG} ";;
    esac
done

opt -load ~/HIPCC_LLVM_4.3.1/lib/lib${pass_so}.so -${pass_name} ${flags} -enable-new-pm=0  ${ll_file_path_name} -o ${target_bc_file_path_name}
