#!/bin/bash

flags="-I/mnt/sdb1/public/yum/env/v1/spack/opt/spack/linux-centos7-zen/gcc-8.5.0/rocsolver-4.3.1-3e5fmmbn6zobkdodtkgew26szbudusdq/include -lrocsolver -L/mnt/sdb1/public/yum/env/v1/spack/opt/spack/linux-centos7-zen/gcc-8.5.0/rocsolver-4.3.1-3e5fmmbn6zobkdodtkgew26szbudusdq/lib -I/mnt/sdb1/public/yum/env/v1/spack/opt/spack/linux-centos7-zen/gcc-8.5.0/rocblas-4.3.1-fhdrqvdjmkc7jrobubifh2mgvg7se4rc/include -lrocblas -L/mnt/sdb1/public/yum/env/v1/spack/opt/spack/linux-centos7-zen/gcc-8.5.0/rocblas-4.3.1-fhdrqvdjmkc7jrobubifh2mgvg7se4rc/lib"

while getopts p:f:n:i:d:t:e: flag
do
    case "${flag}" in
        n) N=${OPTARG};;
        i) it=${OPTARG};;
        p) path=${OPTARG};;
        d) ff=${OPTARG};;
        t) ts=${OPTARG};;
        f) flags="$flags ${OPTARG} ";;
        e) err=${OPTARG};;
    esac
done

funcid_file_path_name="${path}/calledFuncID.txt"
pass_name="tuning"
so_name="TUNING${err}"
ll_path_file="${path}/combine.ll"
new_bc_path_file="${path}/combine-new-new.bc"
bin_o_path_file="${path}/combine-new-new.o"
exec_path_file="${path}/combine-new-new"


opt_source_ll_file=${ll_path_file}


#First time, we need to re-organize the funcid file, so we dont read the file now
echo "First time running pass 2"
/mnt/sdb1/home/mzw/workspace/test_space/llvm_test/useful_command/opt_pass_witharg.sh -n $pass_name -s $so_name -l $opt_source_ll_file -t $new_bc_path_file
/mnt/sdb1/home/mzw/workspace/test_space/llvm_test/useful_command/bc2exe_witharg.sh -b combine-new-new -p ${path} -e combine-new-new
${exec_path_file} $N $it $ff $ts >> ${path}/tuning_log.bat 2>&1

echo "Finish the first time running pass 2"

#Now we have sorted set of runned function id, just loop over it.
#We will run pass 2 one more time than the number of runned function, need to pat attention to it
echo "Begin iterative tuning"
count='0'
while IFS= read -r line; do
    count=$(($count+1))
    /mnt/sdb1/home/mzw/workspace/test_space/llvm_test/useful_command/opt_pass_witharg.sh -n $pass_name -s $so_name -l $opt_source_ll_file -t $new_bc_path_file
    /mnt/sdb1/home/mzw/workspace/test_space/llvm_test/useful_command/bc2exe_witharg.sh -b combine-new-new -p ${path} -e combine-new-new
    ${exec_path_file} $N $it $ff $ts >> ${path}/tuning_log.bat 2>&1
done < "$funcid_file_path_name"

#Last time to check the final result should be changed to -1-1-1 or just be 11x / other combination
/mnt/sdb1/home/mzw/workspace/test_space/llvm_test/useful_command/opt_pass_witharg.sh -n $pass_name -s $so_name -l $opt_source_ll_file -t $new_bc_path_file
/mnt/sdb1/home/mzw/workspace/test_space/llvm_test/useful_command/bc2exe_witharg.sh -b combine-new-new -p ${path} -e combine-new-new
${exec_path_file} $N $it $ff $ts >> ${path}/tuning_log.bat 2>&1

echo "Total '$count' func were tuned"
