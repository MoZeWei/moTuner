#!/bin/bash

read -p "FuncID File Name(with absolute path): " funcid_file_path_name
read -p "Optimize Pass Name: " pass_name
read -p "Pass Shared Library Name(without lib/so): " so_name
read -p "target ll file name to optimize(with path): " ll_path_file
read -p "generated bc file name(with path): " new_bc_path_file
read -p "binary output file name(with path): " bin_o_path_file
read -p "Exec file path(with name): " exec_path_file
read -p "extra flag(with -): " flag


opt_source_ll_file=${ll_path_file}

#First time, we need to re-organize the funcid file, so we dont read the file now
echo "First time running pass 2"
/mnt/sdb1/home/mzw/workspace/test_space/llvm_test/useful_command/opt_pass_witharg.sh -n $pass_name -s $so_name -l $opt_source_ll_file -t $new_bc_path_file -f $flag
/mnt/sdb1/home/mzw/workspace/test_space/llvm_test/useful_command/bc2exe_witharg.sh -b $new_bc_path_file -o $bin_o_path_file -e $exec_path_file
${exec_path_file}

echo "Finish the first time running pass 2"

#Now we have sorted set of runned function id, just loop over it.
#We will run pass 2 one more time than the number of runned function, need to pat attention to it
echo "Begin iterative tuning"
count='0'
while IFS= read -r line; do
    count=$(($count+1))
    /mnt/sdb1/home/mzw/workspace/test_space/llvm_test/useful_command/opt_pass_witharg.sh -n $pass_name -s $so_name -l $opt_source_ll_file -t $new_bc_path_file -f $flag
    /mnt/sdb1/home/mzw/workspace/test_space/llvm_test/useful_command/bc2exe_witharg.sh -b $new_bc_path_file -o $bin_o_path_file -e $exec_path_file
    ${exec_path_file}
done < "$funcid_file_path_name"

#Last time to check the final result should be changed to -1-1-1 or just be 11x / other combination
/mnt/sdb1/home/mzw/workspace/test_space/llvm_test/useful_command/opt_pass_witharg.sh -n $pass_name -s $so_name -l $opt_source_ll_file -t $new_bc_path_file -f $flag
/mnt/sdb1/home/mzw/workspace/test_space/llvm_test/useful_command/bc2exe_witharg.sh -b $new_bc_path_file -o $bin_o_path_file -e $exec_path_file
${exec_path_file}

echo "Total '$count' func were tuned"