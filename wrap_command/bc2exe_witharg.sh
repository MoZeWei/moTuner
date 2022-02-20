#!/bin/bash

flags="-lrocsolver -L/mnt/sdb1/public/yum/env/v1/spack/opt/spack/linux-centos7-zen/gcc-8.5.0/rocsolver-4.3.1-3e5fmmbn6zobkdodtkgew26szbudusdq/lib -lrocblas -L/mnt/sdb1/public/yum/env/v1/spack/opt/spack/linux-centos7-zen/gcc-8.5.0/rocblas-4.3.1-fhdrqvdjmkc7jrobubifh2mgvg7se4rc/lib"


while getopts b:p:f:e: flag
do
    case "${flag}" in
        b) filename=${OPTARG};;
        p) path=${OPTARG};;
        f) flags="$flags ${OPTARG} ";;
        e) exec_filename=${OPTARG};;
    esac
done

clang++ ${path}/${filename}.bc -c -o ${path}/${filename}.o

#This doesn't include -lhipblas
"/mnt/sdb1/home/mzw/HIPCC_LLVM_4.3.1/bin/ld.lld" -z relro --hash-style=gnu ${flags} --eh-frame-hdr -m elf_x86_64 -dynamic-linker /lib64/ld-linux-x86-64.so.2 -o ${path}/${exec_filename} /lib/../lib64/crt1.o /lib/../lib64/crti.o /mnt/sdb1/public/yum/env/v1/spack/opt/spack/linux-centos7-x86_64_v3/gcc-4.8.5/gcc-8.5.0-r653aprkmt2xeqxzz4locto6dy3hxl47/lib/gcc/x86_64-pc-linux-gnu/8.5.0/crtbegin.o -L/mnt/sdb1/public/yum/env/v1/spack/opt/spack/linux-centos7-zen/gcc-8.5.0/hip-4.3.1-3nok6473g7mbkuapaer4tvdu4njjs6hu/lib -L/mnt/sdb1/home/mzw/rocblas_4.3.1/lib -L/mnt/sdb1/public/yum/env/v1/spack/opt/spack/linux-centos7-zen/gcc-8.5.0/llvm-amdgpu-4.3.1-mk5b5zrnseww52hd3dxklqxhv2zxphbd/bin/../lib/clang/13.0.0/lib/linux -L/mnt/sdb1/public/yum/env/v1/spack/opt/spack/linux-centos7-x86_64_v3/gcc-4.8.5/gcc-8.5.0-r653aprkmt2xeqxzz4locto6dy3hxl47/lib/gcc/x86_64-pc-linux-gnu/8.5.0 -L/mnt/sdb1/public/yum/env/v1/spack/opt/spack/linux-centos7-x86_64_v3/gcc-4.8.5/gcc-8.5.0-r653aprkmt2xeqxzz4locto6dy3hxl47/lib/gcc/x86_64-pc-linux-gnu/8.5.0/../../../../lib64 -L/lib/../lib64 -L/usr/lib/../lib64 -L/mnt/sdb1/public/yum/env/v1/spack/opt/spack/linux-centos7-x86_64_v3/gcc-4.8.5/gcc-8.5.0-r653aprkmt2xeqxzz4locto6dy3hxl47/lib/gcc/x86_64-pc-linux-gnu/8.5.0/../../.. -L/mnt/sdb1/public/yum/env/v1/spack/opt/spack/linux-centos7-zen/gcc-8.5.0/llvm-amdgpu-4.3.1-mk5b5zrnseww52hd3dxklqxhv2zxphbd/bin/../lib -L/lib -L/usr/lib -lgcc_s -lgcc -lpthread -lm -lrt ${filename}.o -lrocblas --enable-new-dtags --rpath=/mnt/sdb1/public/yum/env/v1/spack/opt/spack/linux-centos7-zen/gcc-8.5.0/hip-4.3.1-3nok6473g7mbkuapaer4tvdu4njjs6hu/lib:/mnt/sdb1/public/yum/env/v1/spack/opt/spack/linux-centos7-zen/gcc-8.5.0/hip-4.3.1-3nok6473g7mbkuapaer4tvdu4njjs6hu/lib -lamdhip64 -lclang_rt.builtins-x86_64 -lstdc++ -lm -lgcc_s -lgcc -lc -lgcc_s -lgcc /mnt/sdb1/public/yum/env/v1/spack/opt/spack/linux-centos7-x86_64_v3/gcc-4.8.5/gcc-8.5.0-r653aprkmt2xeqxzz4locto6dy3hxl47/lib/gcc/x86_64-pc-linux-gnu/8.5.0/crtend.o /lib/../lib64/crtn.o
