#!/bin/bash

flags=""

while getopts p:f:e: flag
do
    case "${flag}" in
        p) path=${OPTARG};;
        e) err=${OPTARG};;
        f) flags="$flags -f${OPTARG} ";;
    esac
done

original_ll="${path}/combine.ll"
new_bc="${path}/combine-new.bc"
final_bc="${path}/final.bc"
final_exec="${path}/final"
new_exec="${path}/combine-new"


#Pass1
/mnt/sdb1/home/mzw/workspace/test_space/llvm_test/useful_command/work1_pass1.sh -l $original_ll -b $new_bc
#Turn bc to exec
/mnt/sdb1/home/mzw/workspace/test_space/llvm_test/useful_command/bc2exe_witharg.sh -b "combine-new" -p ${path} -e "combine-new"
#Run it
${new_exec}

#Pass2
/mnt/sdb1/home/mzw/workspace/test_space/llvm_test/useful_command/pass2_witharg.sh -p ${path} -e $err

#Final Gen
/mnt/sdb1/home/mzw/workspace/test_space/llvm_test/useful_command/work1_pass3.sh -l $original_ll -b $final_bc
#Gen exec
/mnt/sdb1/home/mzw/workspace/test_space/llvm_test/useful_command/bc2exe_witharg.sh -b "final" -p ${path} -e "final"