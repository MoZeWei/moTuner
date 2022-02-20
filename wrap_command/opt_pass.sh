#!/bin/bash

read -p "pass_name: " pass_name
read -p "pass .so name:(not with the lib and .so) "  pass_so
read -p "ll file path:(including the file name) " file_path_name
read -p "new bc file path:(including the file name)" target_file_path_name
read -p "extra flag(with -): " flag

opt -load ~/HIPCC_LLVM_4.3.1/lib/lib${pass_so}.so -${pass_name} ${flag} -enable-new-pm=0  ${file_path_name} -o ${target_file_path_name}
llvm-dis  ${target_file_path_name}
