#!/bin/bash

flags="-I/mnt/sdb1/public/yum/env/v1/spack/opt/spack/linux-centos7-zen/gcc-8.5.0/rocsolver-4.3.1-3e5fmmbn6zobkdodtkgew26szbudusdq/include -lrocsolver -L/mnt/sdb1/public/yum/env/v1/spack/opt/spack/linux-centos7-zen/gcc-8.5.0/rocsolver-4.3.1-3e5fmmbn6zobkdodtkgew26szbudusdq/lib -I/mnt/sdb1/public/yum/env/v1/spack/opt/spack/linux-centos7-zen/gcc-8.5.0/rocblas-4.3.1-fhdrqvdjmkc7jrobubifh2mgvg7se4rc/include -lrocblas -L/mnt/sdb1/public/yum/env/v1/spack/opt/spack/linux-centos7-zen/gcc-8.5.0/rocblas-4.3.1-fhdrqvdjmkc7jrobubifh2mgvg7se4rc/lib"

while getopts p:f: flag
do
    case "${flag}" in
        p) path=${OPTARG};;
        f) flags="$flags ${OPTARG} ";;
    esac
done

original_ll="${path}/combine.ll"
new_bc="${path}/combine-new.bc"
final_bc="${path}/final.bc"
final_exec="${path}/final"
new_exec="${path}/combine-new"


#Pass1
/mnt/sdb1/home/mzw/workspace/test_space/llvm_test/useful_command/work1_pass1.sh -l $original_ll -b $new_bc #-f ${flags}
#Turn bc to exec
/mnt/sdb1/home/mzw/workspace/test_space/llvm_test/useful_command/bc2exe_witharg.sh -b "combine-new" -p ${path} -e "combine-new"
#Run it
${new_exec}

#Pass2
/mnt/sdb1/home/mzw/workspace/test_space/llvm_test/useful_command/pass2_witharg.sh -p ${path}

#Final Gen
/mnt/sdb1/home/mzw/workspace/test_space/llvm_test/useful_command/work1_pass3.sh -l $original_ll -b $final_bc
#Gen exec
/mnt/sdb1/home/mzw/workspace/test_space/llvm_test/useful_command/bc2exe_witharg.sh -b "final" -p ${path} -e "final"